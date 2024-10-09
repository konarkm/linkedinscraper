import requests
import json
import sqlite3
import sys
from sqlite3 import Error
from bs4 import BeautifulSoup
import time as tm
from itertools import groupby
from datetime import datetime, timedelta, time
import pandas as pd
from urllib.parse import quote
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import openai
import asyncio
import json


def load_config(file_name):
    # Load the config file
    with open(file_name) as f:
        return json.load(f)

def get_with_retry(url, config, retries=3, delay=1):
    # Get the URL with retries and delay
    for i in range(retries):
        try:
            if len(config['proxies']) > 0:
                r = requests.get(url, headers=config['headers'], proxies=config['proxies'], timeout=5)
            else:
                r = requests.get(url, headers=config['headers'], timeout=5)
            return BeautifulSoup(r.content, 'html.parser')
        except requests.exceptions.Timeout:
            print(f"Timeout occurred for URL: {url}, retrying in {delay}s...")
            tm.sleep(delay)
        except Exception as e:
            print(f"An error occurred while retrieving the URL: {url}, error: {e}")
    return None

def transform(soup):
    # Parsing the job card info (title, company, location, date, job_url) from the beautiful soup object
    joblist = []
    try:
        divs = soup.find_all('div', class_='base-search-card__info')
    except:
        print("Empty page, no jobs found")
        return joblist
    for item in divs:
        title = item.find('h3').text.strip()
        company = item.find('a', class_='hidden-nested-link')
        location = item.find('span', class_='job-search-card__location')
        parent_div = item.parent
        entity_urn = parent_div['data-entity-urn']
        job_posting_id = entity_urn.split(':')[-1]
        job_url = 'https://www.linkedin.com/jobs/view/'+job_posting_id+'/'

        date_tag_new = item.find('time', class_ = 'job-search-card__listdate--new')
        date_tag = item.find('time', class_='job-search-card__listdate')
        date = date_tag['datetime'] if date_tag else date_tag_new['datetime'] if date_tag_new else ''
        job_description = ''
        job = {
            'title': title,
            'company': company.text.strip().replace('\n', ' ') if company else '',
            'location': location.text.strip() if location else '',
            'date': date,
            'job_url': job_url,
            'job_description': job_description,
            'applied': 0,
            'hidden': 0,
            'interview': 0,
            'rejected': 0,
            'json': ''
        }
        joblist.append(job)
    return joblist

def transform_job(soup):
    div = soup.find('div', class_='description__text description__text--rich')
    if div:
        # Remove unwanted elements
        for element in div.find_all(['span', 'a']):
            element.decompose()

        # Replace bullet points
        for ul in div.find_all('ul'):
            for li in ul.find_all('li'):
                li.insert(0, '-')

        text = div.get_text(separator='\n').strip()
        text = text.replace('\n\n', '')
        text = text.replace('::marker', '-')
        text = text.replace('-\n', '- ')
        text = text.replace('Show less', '').replace('Show more', '')
        return text
    else:
        return "Could not find Job Description"

def safe_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'en'

def remove_irrelevant_jobs(joblist, config):
    #Filter out jobs based on description, title, and language. Set up in config.json.
    new_joblist = [job for job in joblist if not any(word.lower() in job['job_description'].lower() for word in config['desc_words'])]   
    new_joblist = [job for job in new_joblist if not any(word.lower() in job['title'].lower() for word in config['title_exclude'])] if len(config['title_exclude']) > 0 else new_joblist
    new_joblist = [job for job in new_joblist if any(word.lower() in job['title'].lower() for word in config['title_include'])] if len(config['title_include']) > 0 else new_joblist
    new_joblist = [job for job in new_joblist if safe_detect(job['job_description']) in config['languages']] if len(config['languages']) > 0 else new_joblist
    new_joblist = [job for job in new_joblist if not any(word.lower() in job['company'].lower() for word in config['company_exclude'])] if len(config['company_exclude']) > 0 else new_joblist

    return new_joblist

def remove_duplicates(joblist, config):
    # Remove duplicate jobs in the joblist. Duplicate is defined as having the same title and company.
    joblist.sort(key=lambda x: (x['title'], x['company']))
    joblist = [next(g) for k, g in groupby(joblist, key=lambda x: (x['title'], x['company']))]
    return joblist

def convert_date_format(date_string):
    """
    Converts a date string to a date object. 
    
    Args:
        date_string (str): The date in string format.

    Returns:
        date: The converted date object, or None if conversion failed.
    """
    date_format = "%Y-%m-%d"
    try:
        job_date = datetime.strptime(date_string, date_format).date()
        return job_date
    except ValueError:
        print(f"Error: The date for job {date_string} - is not in the correct format.")
        return None

def create_connection(config):
    # Create a database connection to a SQLite database
    conn = None
    path = config['db_path']
    try:
        conn = sqlite3.connect(path) # creates a SQL database in the 'data' directory
        #print(sqlite3.version)
    except Error as e:
        print(e)

    return conn

def create_table(conn, df, table_name):
    ''''
    # Create a new table with the data from the dataframe
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    print (f"Created the {table_name} table and added {len(df)} records")
    '''
    # Create a new table with the data from the DataFrame
    # Prepare data types mapping from pandas to SQLite
    type_mapping = {
        'int64': 'INTEGER',
        'float64': 'REAL',
        'datetime64[ns]': 'TIMESTAMP',
        'object': 'TEXT',
        'bool': 'INTEGER'
    }
    
    # Prepare a string with column names and their types
    columns_with_types = ', '.join(
        f'"{column}" {type_mapping[str(df.dtypes[column])]}'
        for column in df.columns
    )
    
    # Prepare SQL query to create a new table
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {columns_with_types}
        );
    """
    
    # Execute SQL query
    cursor = conn.cursor()
    cursor.execute(create_table_sql)
    
    # Commit the transaction
    conn.commit()

    # Insert DataFrame records one by one
    insert_sql = f"""
        INSERT INTO "{table_name}" ({', '.join(f'"{column}"' for column in df.columns)})
        VALUES ({', '.join(['?' for _ in df.columns])})
    """
    for record in df.to_dict(orient='records'):
        cursor.execute(insert_sql, list(record.values()))
    
    # Commit the transaction
    conn.commit()

    print(f"Created the {table_name} table and added {len(df)} records")

def update_table(conn, df, table_name):
    # Update the existing table with new records.
    df_existing = pd.read_sql(f'select * from {table_name}', conn)

    # Create a dataframe with unique records in df that are not in df_existing
    df_new_records = pd.concat([df, df_existing, df_existing]).drop_duplicates(['title', 'company', 'date'], keep=False)

    # If there are new records, append them to the existing table
    if len(df_new_records) > 0:
        df_new_records.to_sql(table_name, conn, if_exists='append', index=False)
        print (f"Added {len(df_new_records)} new records to the {table_name} table")
    else:
        print (f"No new records to add to the {table_name} table")

def table_exists(conn, table_name):
    # Check if the table already exists in the database
    cur = conn.cursor()
    cur.execute(f"SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    if cur.fetchone()[0]==1 :
        return True
    return False

def job_exists(df, job):
    # Check if the job already exists in the dataframe
    if df.empty:
        return False
    #return ((df['title'] == job['title']) & (df['company'] == job['company']) & (df['date'] == job['date'])).any()
    #The job exists if there's already a job in the database that has the same URL
    return ((df['job_url'] == job['job_url']).any() | (((df['title'] == job['title']) & (df['company'] == job['company']) & (df['date'] == job['date'])).any()))

def get_jobcards(config):
    #Function to get the job cards from the search results page
    all_jobs = []
    for k in range(0, config['rounds']):
        for query in config['search_queries']:
            keywords = quote(query['keywords']) # URL encode the keywords
            location = quote(query['location']) # URL encode the location
            for i in range (0, config['pages_to_scrape']):
                url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={keywords}&location={location}&f_TPR=&f_WT={query['f_WT']}&geoId=&f_TPR={config['timespan']}&start={25*i}"
                soup = get_with_retry(url, config)
                jobs = transform(soup)
                all_jobs = all_jobs + jobs
                print("Finished scraping page:", url)
    print ("Total job cards scraped:", len(all_jobs))
    all_jobs = remove_duplicates(all_jobs, config)
    print ("Total job cards after removing duplicates (from all the scraped jobs):", len(all_jobs))
    all_jobs = remove_irrelevant_jobs(all_jobs, config)
    print ("Total job cards after removing irrelevant jobs (based on config filters):", len(all_jobs))
    return all_jobs

def find_new_jobs(all_jobs, conn, config):
    # From all_jobs, find the jobs that are not already in the database. Function checks both the jobs and filtered_jobs tables.
    jobs_tablename = config['jobs_tablename']
    filtered_jobs_tablename = config['filtered_jobs_tablename']
    jobs_db = pd.DataFrame()
    filtered_jobs_db = pd.DataFrame()    
    if conn is not None:
        if table_exists(conn, jobs_tablename):
            query = f"SELECT * FROM {jobs_tablename}"
            jobs_db = pd.read_sql_query(query, conn)
        if table_exists(conn, filtered_jobs_tablename):
            query = f"SELECT * FROM {filtered_jobs_tablename}"
            filtered_jobs_db = pd.read_sql_query(query, conn)

    new_joblist = [job for job in all_jobs if not job_exists(jobs_db, job) and not job_exists(filtered_jobs_db, job)]
    return new_joblist

async def get_response(description, client):
    # Get the response from OpenAI based on the job description

    # Set up the system prompt
    system_prompt = """You are a an assistant who reads job descriptions and is designed to output JSON with values depending on the following. Do not vary from the format. Focus on the requirements/required qualifications/minimum requirements section, and disregard the rest (including things like the preferred qualifications sections) There are 2 parts in the output:
     1. key: "education", value: 0 for no college experience, 1 for bachelor's degree (BS/BA), 2 for Master's degree, 3 for PhD. This value should be set based on the minimum required in the description.
     2. key: "experience", value: integer corresponding to the minimum years of experience required. For example, "2+" would result in the value being 2, "at least 3" would be 3, and if experience is not mentioned it would be 0.

     If you are unable to figure out any of the values, set them to 0."""
    
    # Set up the user prompt
    user_prompt = str(description)

    # Get the response from OpenAI in JSON format
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response = completion.choices[0].message.content
    except Exception as e:
        print(f"Error connecting to OpenAI: {e}")
        # response = json.dumps({"education": 0, "experience": 0})
    
    return response

async def update_dataframe(df, client, config):
    # Update the dataframe with the JSON response from OpenAI, and hide jobs based on the user's information (currently education and experience).

    # Filter out rows where the JSON column is null
    filtered_df = df[~df['json'].isnull()].copy()

    # Create a list of tasks to get the response from OpenAI
    tasks = [get_response(description, client) for description in filtered_df['job_description']]

    # Run the tasks concurrently
    responses = await asyncio.gather(*tasks)

    # Set the JSON response in the dataframe
    filtered_df['json'] = responses

    # Get the user's info from the config file
    user_education = int(config["user_info"]["education"])
    user_experience = int(config["user_info"]["experience"])

    # Update the hidden column based on the JSON response in filtered_df.
    # Specifically, if the user's education and experience are less than or equal to the values in the JSON response, the job is hidden.
    # Note: json.loads only here and not stored in the df because sqlite3 doesn't support JSON columns, and so the json column is saved as a JSON string.
    filtered_df['hidden'] = filtered_df['json'].apply(
        lambda x: 1 if json.loads(x).get('education') > user_education or json.loads(x).get('experience') > user_experience else 0
    )

    # Print the number of jobs to add, the number of jobs hidden by AI, and the final number of jobs to add
    filtered_df_rows = len(filtered_df)
    hidden_rows = len(filtered_df[filtered_df['hidden'] == 1])
    # print("Total number of jobs to add:", filtered_df_rows)
    print("Number of jobs that were hidden by AI:", hidden_rows)
    print("Final number of jobs to add:", filtered_df_rows - hidden_rows)

    # Update the main dataframe with values from filtered_df
    df.loc[filtered_df.index, ['json', 'hidden']] = filtered_df[['json', 'hidden']]

    return df

async def main(config_file):
    start_time = tm.perf_counter()
    job_list = []

    config = load_config(config_file)
    client = openai.AsyncOpenAI(api_key=config["OpenAI_API_KEY"])
    jobs_tablename = config['jobs_tablename'] # name of the table to store the "approved" jobs
    filtered_jobs_tablename = config['filtered_jobs_tablename'] # name of the table to store the jobs that have been filtered out based on description keywords (so that in future they are not scraped again)
    #Scrape search results page and get job cards. This step might take a while based on the number of pages and search queries.
    all_jobs = get_jobcards(config)
    conn = create_connection(config)
    #filtering out jobs that are already in the database
    all_jobs = find_new_jobs(all_jobs, conn, config)
    print ("Total new jobs found after removing those that are already in the database:", len(all_jobs))

    if len(all_jobs) > 0:

        for job in all_jobs:
            job_date = convert_date_format(job['date'])
            job_date = datetime.combine(job_date, time())
            #if job is older than a week, skip it
            if job_date < datetime.now() - timedelta(days=config['days_to_scrape']):
                continue
            print('Found new job: ', job['title'], 'at ', job['company'], job['job_url'])
            desc_soup = get_with_retry(job['job_url'], config)
            job_description = transform_job(desc_soup)
            if job_description == "Could not find Job Description":
                continue
            job['job_description'] = job_description
            language = safe_detect(job['job_description'])
            if language not in config['languages']:
                print('Job description language not supported: ', language)
                continue
            job_list.append(job)
        #Final check - removing jobs based on job description keywords words from the config file
        jobs_to_add = remove_irrelevant_jobs(job_list, config)
        print ("Total jobs to screen with AI:", len(jobs_to_add))
        #Create a list for jobs removed based on job description keywords - they will be added to the filtered_jobs table
        filtered_list = [job for job in job_list if job not in jobs_to_add]
        df = pd.DataFrame(jobs_to_add)
        df = await update_dataframe(df, client, config)
        df_filtered = pd.DataFrame(filtered_list)
        df['date_loaded'] = datetime.now()
        df_filtered['date_loaded'] = datetime.now()
        df['date_loaded'] = df['date_loaded'].astype(str)
        df_filtered['date_loaded'] = df_filtered['date_loaded'].astype(str)        
        
        if conn is not None:
            #Update or Create the database table for the job list
            if table_exists(conn, jobs_tablename):
                update_table(conn, df, jobs_tablename)
            else:
                create_table(conn, df, jobs_tablename)
                
            #Update or Create the database table for the filtered out jobs
            if table_exists(conn, filtered_jobs_tablename):
                update_table(conn, df_filtered, filtered_jobs_tablename)
            else:
                create_table(conn, df_filtered, filtered_jobs_tablename)
        else:
            print("Error! cannot create the database connection.")
        
        df.to_csv('linkedin_jobs.csv', mode='a', index=False, encoding='utf-8')

        df_filtered.to_csv('linkedin_jobs_filtered.csv', mode='a', index=False, encoding='utf-8')
    else:
        print("No jobs found")
    
    end_time = tm.perf_counter()
    print(f"Scraping finished in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    config_file = 'config.json'  # default config file
    if len(sys.argv) == 2:
        config_file = sys.argv[1]
        
    asyncio.run(main(config_file))
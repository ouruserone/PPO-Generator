import pinecone
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import os
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import requests
import json

import pyodbc
import traceback
import re

import smtplib,ssl
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders
import datetime
import time
from datetime import datetime, date
import os
import errno
import json
import openai

from langchain.chains.qa_with_sources import load_qa_with_sources_chain



################Create Log file##################################
filename = "./logs/"+str(date.today())+".log"
if not os.path.exists(os.path.dirname(filename)):
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

################Create Error Log file##################################
error_filename = "./Error_logs/Error_"+str(date.today())+".log"
if not os.path.exists(os.path.dirname(error_filename)):
    try:
        os.makedirs(os.path.dirname(error_filename))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


################Create ChatGPT Log file##################################
chatgpt_filename_log = "./ChatGpt_logs/ChatGpt_Logs_"+str(date.today())+".log"
if not os.path.exists(os.path.dirname(chatgpt_filename_log)):
    try:
        os.makedirs(os.path.dirname(chatgpt_filename_log))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


error = None


def writeLog(info):
    f = open(filename, "a", encoding='utf-8')
    f.write("\n"+str(datetime.now())+"\t"+info)
    f.close()


def ErrorwriteLog(info):
    f = open(error_filename, "a", encoding='utf-8')
    f.write("\n"+str(datetime.now())+"\t"+info)
    f.close()


def ChatGptwriteLog(info):
    f = open(chatgpt_filename_log, "a", encoding='utf-8')
    f.write("\n"+str(datetime.now())+"\t"+info)
    f.close()


def send_mail(send_from,send_to,subject,text):
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = ", ".join(send_to)
    msg['Date'] = formatdate(localtime = True)
    msg['Subject'] = subject
    msg.attach(MIMEText(text))

    #context = ssl.SSLContext(ssl.PROTOCOL_SSLv3)
    #SSL connection only working on Python 3+
    smtp = smtplib.SMTP('smtp.adfactorspr.com',587)
    smtp.starttls()
    # # if isTls:
    #       smtp.starttls()

    smtp.login('noreply@adfactorspr.com','N0r3ply#')
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()


def send_error(error_trace, topic_id, topic_name, emailids):
    msg_str_error = "Topic ID:- {} \n Topic Name:- {} \n".format(str(topic_id), topic_name)
    ErrorwriteLog(str(msg_str_error))

    error_str = 'Error for Topic Id {}'.format(str(topic_id)) + '\n' + str(error_trace)
    print(error_str)
    writeLog(str(error_str))
    ErrorwriteLog(str(error_str))
    record_processed_datetime = datetime.now()

    is_error = 1
    if len(str(error_str)) < 10000:
        error_reason = str(error_str)
    else:
        error_reason = "Check Error logs"

    message = 'Hi Team,\n\nError Generated in PPO Generator Service for Topic Id '+ str(topic_id) +'\n '+ str(error_reason) +' \n\n\n\nThanks,\n'
    send_mail(send_from='noreply@adfactorspr.com',send_to=emailids,subject='Error in PPO Generator Service' ,text=str(message))



def get_similar_docs_id(vectorstore, topic_name, sub_topic, index, openai_api_key):
    
    # vectorize query
    openai.api_key = openai_api_key

    start_date_str = '25-07-23 10:00:00'
    start_date_object = datetime.strptime(start_date_str, '%d-%m-%y %H:%M:%S')
    start_date_timestamp = start_date_object.timestamp()

    end_date_str = '26-07-23 10:00:00'
    end_date_object = datetime.strptime(end_date_str, '%d-%m-%y %H:%M:%S')
    end_date_timestamp = end_date_object.timestamp()

    if sub_topic:
        input_query = sub_topic
    else:
        input_query = topic_name

    try:
        query_vector = openai.Embedding.create(
            input=input_query,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]
    except Exception as e:
        print(str(traceback.print_exc()))
        log_str = f"Error calling OpenAI Embedding API for topic:- {topic_name}" + '\n' + str(traceback.print_exc())
        print(log_str)
        ErrorwriteLog(log_str)
        writeLog(log_str)

    # query with pinecone
    search_response = []
    search_response = index.query(top_k=5, vector=query_vector, include_metadata=True,
                                  filter={'$and': [
                                    {'timestamp': {'$gte': start_date_timestamp}},
                                    {'timestamp': {'$lte': end_date_timestamp}}
                                ]})

    log_str = f"Retriever query for Pinecone is:- {input_query}"
    print(log_str)
    writeLog(log_str)


    if 'matches' in search_response:
        retrieved_docs_list = search_response['matches']
        similar_id_list = []
        similar_id_with_score = {}
        for retrieved_doc in retrieved_docs_list:
            metadata = retrieved_doc['metadata']
            # print(metadata)

            if 'id' in metadata:
                id = metadata['id']
                print('id:- ', id)
                similar_id_list.append(id)
                similar_id_with_score[id] = retrieved_doc['score']

        log_str = f"Retriever data from Pinecone is:- {similar_id_with_score}"
        print(log_str)
        writeLog(log_str)
        

    return retrieved_docs_list, similar_id_list


def get_relevant_articles(topic_id, topic, sub_topic, article_dict, openai_api_key, emailids):
    chat_gpt_error = False
    chat_gpt_error_str = ''
    relevant_article_list = []
    if article_dict:
        article_dict_keys_list = article_dict.keys()
        for article_dict_keys in article_dict_keys_list:
            article_id = article_dict_keys
            article = article_dict[article_dict_keys]
            retry_count = 0
            loop_continue = True
            while loop_continue:
                try:
                    URL = "https://api.openai.com/v1/chat/completions"

                    api_key = openai_api_key

                    messages = []

                    ### Is this article about {topic}?\n \

                    if sub_topic:
                        prompt = f'''
                                You would be provided with article in the text delimited with triple backticks.\n\
                                Can you please answer below question.\n \
                                
                                Is this article about {sub_topic}?\n \
                                
                                Article is  ```{article}```\n \
                                Can you please provide output in Valid JSON like:-\n \
                                {{
                                    "Article Related": "Yes or No",
                                }}
                        '''
                    else:
                        prompt = f'''
                                You would be provided with article in the text delimited with triple backticks.\n\
                                Can you please answer below question.\n \
                                
                                Is this article about {topic}?\n \
                                
                                Article is  ```{article}```\n \
                                Can you please provide output in Valid JSON like:-\n \
                                {{
                                    "Article Related": "Yes or No",
                                }}
                                '''

                    model = "gpt-3.5-turbo"
                    # model = "gpt-4"

                    messages.append({"role": "user", "content": prompt })
                    print('messages:- ', messages)

                    payload = {
                    "model": model,
                    # "messages": [{"role": "user", "content": content}],
                    "messages": messages,
                    "temperature" : 0.0,
                    "top_p":1.0,
                    "n" : 1,
                    "stream": False,
                    "presence_penalty":0,
                    "frequency_penalty":0,
                    }

                    headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                    }

                    response = requests.post(URL, headers=headers, json=payload, stream=False)
                    # print('response:- ', response.text)
                    # print('response:- ', type(response.text))

                    
                    print('response.status_code:- ', response.status_code)

                    if response.status_code == 200:
                        json_response = json.loads(response.text)
                        output_response = json_response['choices'][0]['message']['content']

                        print('response_id:- ', json_response['id'])
                        print('object:- ', json_response['object'])
                        print('created:- ', json_response['created'])
                        print('model:- ', json_response['model'])
                        print('total_tokens:- ', json_response['usage']['total_tokens'])
                        print('response_content:- ', json_response['choices'][0]['message']['content'])

                        if sub_topic:
                            ChatGptwriteLog(str("ChatGpt relevant articles for topic {} and sub topic {} Input for Article ID {} is:- \n {}\n").format(str(topic), str(sub_topic), str(article_id), prompt))
                            ChatGptwriteLog(str("ChatGpt relevant articles for topic {} and sub topic {} Response for Article ID {} is:- \n {}\n").format(str(topic), str(sub_topic), str(article_id), output_response))
                        else:
                            ChatGptwriteLog(str("ChatGpt relevant articles for topic {} Input for Article ID {} is:- \n {}\n").format(str(topic), str(article_id), prompt))
                            ChatGptwriteLog(str("ChatGpt relevant articles for topic {} Response for Article ID {} is:- \n {}\n").format(str(topic), str(article_id), output_response))

                        output_json = json.loads(output_response)
                        if str(output_json['Article Related']).lower() == 'yes':
                            relevant_article_list.append(article_id)

                        loop_continue = False
                    else:
                        retry_count += 1
                        output_response = {}
                        chat_gpt_error = True
                        if retry_count >= 3:
                            loop_continue = False
                            chat_gpt_error_str = ("There is Error in ChatGPT Response for relevant Article ID {}. Response Stauts code is {}").format(str(article_id), str(response.status_code))
                            send_error(chat_gpt_error_str, topic_id, topic, emailids)
                except:
                    retry_count += 1
                    chat_gpt_error = True
                    chat_gpt_error_str_trace = str(traceback.print_exc())
                    if retry_count >= 3:
                        loop_continue = False
                        chat_gpt_error_str = ("There is Error in ChatGPT Response for relevant Article ID {}.\n {}\n").format(str(article_id), str(chat_gpt_error_str_trace))
                        send_error(chat_gpt_error_str, topic_id, topic, emailids)

                time.sleep(1)

    return relevant_article_list


def recheck_for_client(topic_id, topic_name, sub_topic, client, article_dict, openai_api_key, emailids):
    chat_gpt_error = False
    chat_gpt_error_str = ''
    article_for_client_list = []
    if article_dict:
        article_dict_keys_list = article_dict.keys()
        for article_dict_keys in article_dict_keys_list:
            article_id = article_dict_keys
            article = article_dict[article_dict_keys]
            retry_count = 0
            loop_continue = True
            while loop_continue:
                try:
                    URL = "https://api.openai.com/v1/chat/completions"

                    api_key = openai_api_key

                    messages = []

                    ## Is this article about {client}?\n \
                    ## Does article has passing mention about {client}?\n \

                    prompt = f'''
                    You would be provided with article in the text delimited with triple backticks.\n\
                    Can you please answer below questions.\n \
                    
                    Is this article about {client}?\n \
                    
                    Article is  ```{article}```\n \
                    Can you please provide output in Valid JSON like:-\n \
                    {{
                        "Article about client": "Yes or No",
                    }}
                    '''

                    model = "gpt-3.5-turbo"
                    # model = "gpt-4"

                    messages.append({"role": "user", "content": prompt })
                    print('messages:- ', messages)

                    payload = {
                    "model": model,
                    # "messages": [{"role": "user", "content": content}],
                    "messages": messages,
                    "temperature" : 0.0,
                    "top_p":1.0,
                    "n" : 1,
                    "stream": False,
                    "presence_penalty":0,
                    "frequency_penalty":0,
                    }

                    headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                    }

                    response = requests.post(URL, headers=headers, json=payload, stream=False)
                    # print('response:- ', response.text)
                    # print('response:- ', type(response.text))

                    
                    print('response.status_code:- ', response.status_code)

                    if response.status_code == 200:
                        json_response = json.loads(response.text)
                        output_response = json_response['choices'][0]['message']['content']

                        print('response_id:- ', json_response['id'])
                        print('object:- ', json_response['object'])
                        print('created:- ', json_response['created'])
                        print('model:- ', json_response['model'])
                        print('total_tokens:- ', json_response['usage']['total_tokens'])
                        print('response_content:- ', json_response['choices'][0]['message']['content'])

                        if sub_topic:
                            ChatGptwriteLog(str("ChatGpt client recheck for topic {} and sub topic {} articles Input for Article ID {} is:- \n {}\n").format(str(topic_name), str(sub_topic), str(article_id), prompt))
                            ChatGptwriteLog(str("ChatGpt client recheck for topic {} and sub topic {} articles Response for Article ID {} is:- \n {}\n").format(str(topic_name), str(sub_topic), str(article_id), output_response))
                        else:
                            ChatGptwriteLog(str("ChatGpt client recheck for topic {} articles Input for Article ID {} is:- \n {}\n").format(str(topic_name), str(article_id), prompt))
                            ChatGptwriteLog(str("ChatGpt client recheck for topic {} articles Response for Article ID {} is:- \n {}\n").format(str(topic_name), str(article_id), output_response))

                        output_json = json.loads(output_response)
                        if str(output_json['Article about client']).lower() == 'no':
                            article_for_client_list.append(article_id)

                        loop_continue = False
                    else:
                        retry_count += 1
                        output_response = {}
                        chat_gpt_error = True
                        if retry_count >= 3:
                            loop_continue = False
                            chat_gpt_error_str = ("There is Error in ChatGPT Response for client recheck Article ID {}. Response Stauts code is {}").format(str(article_id), str(response.status_code))
                            send_error(chat_gpt_error_str, topic_id, topic_name, emailids)
                except:
                    retry_count += 1
                    chat_gpt_error = True
                    chat_gpt_error_str_trace = str(traceback.print_exc())
                    if retry_count >= 3:
                        loop_continue = False
                        chat_gpt_error_str = ("There is Error in ChatGPT Response for client recheck Article ID {}.\n {}\n").format(str(article_id), str(chat_gpt_error_str_trace))
                        send_error(chat_gpt_error_str, topic_id, topic_name, emailids)

                time.sleep(1)

    return article_for_client_list


def get_best_id_source_chain(topic_id, topic_name, sub_topic, client_name, client_desc, client_article_details_dict, emailids):

    langchain_best_match_found = False
    langchain_best_article_id = None

    retrieved_docs_list = []

    for article_id in client_article_details_dict.keys():
        article_headline = client_article_details_dict[article_id]["headline"]
        article_text = client_article_details_dict[article_id]["text"]
        retrieved_docs_list.append(Document(page_content=str(article_text), metadata={"id":str(article_id), "headline": str(article_headline), "source": str(article_id)}))
    
        # retrieved_docs_list.append(doc_obj)

    llm = OpenAI(temperature=0.0)
    #green financing
    if sub_topic:
        query = f"""What is most relevant text about {sub_topic} suitable for news jacketing exercise of {client_name}? \n
                    {client_desc} \n
                    If there is no article suitable for news jacketing exercise of {client_name} then simply respond \"No best match.\"
                """
    else:
        query = f"""What is most relevant text about {topic_name} suitable for news jacketing exercise of {client_name}? \n
                    {client_desc} \n
                    If there is no article suitable for news jacketing exercise of {client_name} then simply respond \"No best match.\"
                """
    
    chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
    output_response = chain({"input_documents": retrieved_docs_list, "question": query}, return_only_outputs=True)

    log_str = f"Langchain Sources Chain query is:- {query}"
    print(log_str)
    writeLog(log_str)

    if 'output_text' in output_response:
        output_response_text = output_response['output_text']
        print('Output:- ', output_response_text)
        if 'no best match' not in output_response_text.lower():
            result = re.search("\\nSOURCES:\s\d+", output_response_text)
            if result:
                print(result[0])
                source_list = re.search("\d+", result[0])
                if source_list:
                    langchain_best_article_id = str(source_list[0])
                    langchain_best_match_found = True
                    print('langchain_best_article_id:- ', langchain_best_article_id)
        else:
            print('No best article found in langchain chains')
    else:
        print('No output_text found in langchain chains')

    if sub_topic:
        log_str = f"Langchain Response found for topic {topic_name} and for sub topic {sub_topic}:- {str(output_response)}"
    else:
        log_str = f"Langchain Response found for topic {topic_name}:- {str(output_response)}"
    print(log_str)
    writeLog(log_str)

    return langchain_best_match_found, langchain_best_article_id



def client_media_opportunities(topic_id, topic_name, sub_topic, client, article_dict, opportunity_name, openai_api_key, emailids, client_desc):
    chat_gpt_error = False
    chat_gpt_error_str = ''
    media_opportunities = None
    article_summary = None
    quote= None
    articles_str = ''
    if article_dict:
        article_dict_keys_list = article_dict.keys()
        for article_dict_keys in article_dict_keys_list:
            article_id = article_dict_keys
            article = article_dict[article_dict_keys]
            # articles_str += f"```Article id: {article_id} \n Article text: {article}``` \n \n"
            articles_str += f"```Article text: {article}``` \n \n"
        retry_count = 0
        loop_continue = True
        while loop_continue:
            try:
                URL = "https://api.openai.com/v1/chat/completions"

                api_key = openai_api_key

                messages = []

                question_prompt = f'''
                You would be provided with article which is text delimited by triple backticks..\n \n \
                
                Your task is to help a communications team of {client}. \n 

                {client_desc} \n \n\
                
                Your task is to perform following actions. \n \n
                1 - Give me an idea about how a company spokesperson can be featured as a media opportunity for an article. \n \n

                2 - Provide a sample quote that can be included in the article.

                3 - Prepare a summary of article provided in text delimited by triple backticks.\n \n

                4 - Prepare output in JSON like:-\n \
                {{
                    "quote": "quote",
                    "article summary": "article summary",
                    "media opportunity": "media opportunity"
                }}

                Text:
                {articles_str}
                '''

                # prompt = article_prompt + question_prompt
                prompt = question_prompt

                model = "gpt-3.5-turbo"
                # model = "gpt-4"

                messages.append({"role": "user", "content": prompt })
                print('messages:- ', messages)



                payload = {
                "model": model,
                # "messages": [{"role": "user", "content": content}],
                "messages": messages,
                "temperature" : 0.0,
                "top_p":1.0,
                "n" : 1,
                "stream": False,
                "presence_penalty":0,
                "frequency_penalty":0,
                }

                headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
                }

                response = requests.post(URL, headers=headers, json=payload, stream=False)
                # print('response:- ', response.text)
                # print('response:- ', type(response.text))

                
                print('response.status_code:- ', response.status_code)

                if response.status_code == 200:
                    json_response = json.loads(response.text)
                    output_response = json_response['choices'][0]['message']['content']

                    print('response_id:- ', json_response['id'])
                    print('object:- ', json_response['object'])
                    print('created:- ', json_response['created'])
                    print('model:- ', json_response['model'])
                    print('total_tokens:- ', json_response['usage']['total_tokens'])
                    print('response_content:- ', json_response['choices'][0]['message']['content'])

                    if sub_topic:
                        ChatGptwriteLog(str("ChatGpt for media opportunitiy {} Input for Topic {} and sub topic {} is:- \n {}\n").format(str(opportunity_name), str(topic_name), str(sub_topic), prompt))
                        ChatGptwriteLog(str("ChatGpt for media opportunitiy {} Response for Topic {} and sub topic {} is:- \n {}\n").format(str(opportunity_name), str(topic_name), str(sub_topic), output_response))
                    else:
                        ChatGptwriteLog(str("ChatGpt for media opportunitiy {} Input for Topic {} is:- \n {}\n").format(str(opportunity_name), str(topic_name), prompt))
                        ChatGptwriteLog(str("ChatGpt for media opportunitiy {} Response for Topic {} is:- \n {}\n").format(str(opportunity_name), str(topic_name), output_response))

                    output_json = json.loads(output_response)
                    if ('quote' in output_json) and ('media opportunity' in output_json) and ('article summary' in output_json):
                        quote = str(output_json['quote'])
                        media_opportunities = str(output_json['media opportunity'])
                        article_summary = str(output_json['article summary'])

                    loop_continue = False
                else:
                    retry_count += 1
                    output_response = {}
                    chat_gpt_error = True
                    if retry_count >= 3:
                        loop_continue = False
                        chat_gpt_error_str = ("There is Error in ChatGPT Response for media opportunities articles for Article ID {}. Response Stauts code is {}").format(str(article_id), str(response.status_code))
                        send_error(chat_gpt_error_str, topic_id, topic_name, emailids)
            except:
                retry_count += 1
                chat_gpt_error = True
                chat_gpt_error_str_trace = str(traceback.print_exc())
                if retry_count >= 3:
                    loop_continue = False
                    chat_gpt_error_str = ("There is Error in ChatGPT Response for media opportunities articles for Article ID {}.\n {}\n").format(str(article_id), str(chat_gpt_error_str_trace))
                    send_error(chat_gpt_error_str, topic_id, topic_name, emailids)

    return media_opportunities, article_summary, quote

def main():
    emailids=['shubham.joshi25007@gmail.com']
    writeLog('\n\n Service Started \n')
    ## Shubham EDIT
    ##################EmailScanner Datbase Details
    server='203.115.122.28'
    database='ADF_Sahadev'
    UID='adfactorsorm'
    PWD='c0ld1*5'

    conn = pyodbc.connect('DRIVER={SQL Server};SERVER='+server+';DATABASE='+database+';UID='+UID+';PWD='+PWD)
    cursor = conn.cursor()

    pinecone_api_key = "a91124d9-d6fa-483e-8343-fc0234f321fe"       #### ouruserone@gmail.com
    pinecone_env = "us-west4-gcp"                                   #### ouruserone@gmail.com
    # pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENV"])
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

    openai_api_key = "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"    ##### Kiran Sir

    os.environ["OPENAI_API_KEY"] = openai_api_key   
    embeddings = OpenAIEmbeddings()

    # index_name="ppo-data-test-2"
    index_name="ppo-data"
    index = pinecone.Index(index_name)
    text_field = "text"


    vectorstore = Pinecone(
        index, embeddings.embed_query, text_field
    )


    get_article_query = "exec [USP_PPO_FetchTopics]"

    cursor.execute(get_article_query)
    
    topic_list = cursor.fetchall()

    print(len(topic_list))
    
    for row in topic_list:
        print(row[0])
        topic_id = row[0]
        topic_name = row[1]

        log_str = f'\n ********************** New Topic:- {topic_name} ********************** \n'
        print(log_str)
        writeLog(log_str)

        try:
            if topic_name and topic_id:
                get_client_query = "exec [USP_PPO_FetchApplicableClients] @TopicID=?"
                values = (int(topic_id))
                cursor.execute(get_client_query, values)

                client_list = cursor.fetchall()
                print(len(client_list))
                
                if len(client_list) > 0:
                    for row in client_list:
                        print(row[0])
                        client_id = row[0]
                        client_name = row[1]
                        client_desc = row[2]
                        try:
                            sub_topic = row[3]
                            # sub_topic = 'Green Financing'
                        except:
                            sub_topic = None

                        if sub_topic:
                            log_str = f'\n ********************** New SubTopic:- {sub_topic} ********************** \n'
                            print(log_str)
                            writeLog(log_str)
                        
                        log_str = f'\n ------------------ Client:- {client_name} ------------------ \n'
                        print(log_str)
                        writeLog(log_str)

                        retrieved_docs_list, similar_id_list = get_similar_docs_id(vectorstore, topic_name, sub_topic, index, openai_api_key)
                        print(similar_id_list)
                        if sub_topic:
                            log_str = f"Pinecone retrieved Articles Id {str(similar_id_list)} found for sub topic {sub_topic} and topic {topic_name}"
                        else:
                            log_str = f"Pinecone retrieved Articles Id {str(similar_id_list)} found for topic {topic_name}"
                        print(log_str)
                        writeLog(log_str)

                        article_dict = {}
                        article_details_dict = {}
                        relevant_article_list = []

                        if similar_id_list:
                            fetch_article_query = "exec [USP_PPO_FetchArticlesByArticleIds] @ArticleIds=?"
                            similar_id_list = [int(id) for id in similar_id_list]
                            similar_id_list_str = ','.join(map(str, similar_id_list))
                            similar_id_values = (similar_id_list_str)
                            cursor.execute(fetch_article_query, similar_id_values)
                            retrieved_articles_list = cursor.fetchall()
                            print(len(retrieved_articles_list))
                            for row in retrieved_articles_list:
                                print(row[0])
                                article_id = row[0]
                                entity_id = row[1]
                                article_title = row[2]
                                article_text = row[3]
                                article_title_text = row[4]
                                article_date = row[5]

                                article_dict[article_id] = article_text

                                article_details_dict[article_id] = {"headline": article_title,
                                                                    "text": article_text}

                        if article_dict:
                            relevant_article_list = get_relevant_articles(topic_id, topic_name, sub_topic, article_dict, openai_api_key, emailids)
                            if sub_topic:
                                log_str = f"Relevant retrieved Articles Id {str(relevant_article_list)} found for topic {topic_name} and for sub topic {sub_topic}"
                            else:
                                log_str = f"Relevant retrieved Articles Id {str(relevant_article_list)} found for topic {topic_name}"
                            print(log_str)
                            writeLog(log_str)

                            if relevant_article_list:
                                ap_id = 0
                                relevant_article_dict = {}
                                for article_id in relevant_article_list:
                                    relevant_article_dict[article_id] = article_dict[article_id]

                                article_for_client_list = recheck_for_client(topic_id, topic_name, sub_topic, client_name, relevant_article_dict, openai_api_key, emailids)
                                if sub_topic:
                                    log_str = f"Client check retrieved Articles Id {str(article_for_client_list)} found for topic {topic_name} and for sub topic {sub_topic}"
                                else:
                                    log_str = f"Client check retrieved Articles Id {str(article_for_client_list)} found for topic {topic_name}"
                                print(log_str)
                                writeLog(log_str)

                                if article_for_client_list:
                                    client_article_details_dict = {}
                                    for article_id in article_for_client_list:
                                        client_article_details_dict[article_id] = article_details_dict[article_id]

                                    langchain_best_match_found, langchain_best_article_id = get_best_id_source_chain(topic_id, topic_name, sub_topic, client_name, client_desc, client_article_details_dict, emailids)
                                    if sub_topic:
                                        log_str = f"Langchain Best Articles Id {str(langchain_best_article_id)} found for topic {topic_name} and for sub topic {sub_topic}"
                                    else:
                                        log_str = f"Langchain Best Articles Id {str(langchain_best_article_id)} found for topic {topic_name}"
                                    print(log_str)
                                    writeLog(log_str)

                                    if langchain_best_match_found:
                                        get_opportunity_query = "exec [USP_PPO_FetchOpportunities] @TopicID=?, @ClientID=?"
                                        values = (int(topic_id), int(client_id))

                                        cursor.execute(get_opportunity_query, values)
                                        
                                        opportunity_list = cursor.fetchall()

                                        print(len(opportunity_list))
                                        
                                        if len(opportunity_list) > 0:
                                            for row in opportunity_list:
                                                print(row[0])
                                                opportunity_id = row[0]
                                                opportunity_name = row[1]

                                                # print('article_dict[langchain_best_article_id]:- ', article_dict[int(langchain_best_article_id)])
                                                langchain_article_dict = {}
                                                langchain_article_dict[int(langchain_best_article_id)] = article_dict[int(langchain_best_article_id)]
                                                # langchain_article_dict.add(langchain_best_article_id, article_dict[langchain_best_article_id])

                                                media_opportunities, article_summary, quote = client_media_opportunities(topic_id, topic_name, sub_topic, client_name, langchain_article_dict, opportunity_name, openai_api_key, emailids, client_desc)
                                                if sub_topic:
                                                    log_str = f"Best retrieved Articles Id {str(langchain_best_article_id)} found for topic {topic_name} and for sub topic {sub_topic} for client {client_name}"
                                                else:
                                                    log_str = f"Best retrieved Articles Id {str(langchain_best_article_id)} found for topic {topic_name} for client {client_name}"
                                                print(log_str)
                                                writeLog(log_str)

                                                if langchain_best_match_found:
                                                    answer_insert_query = "exec [USP_AnswerPPO_Insert] @APID=?, @TopicID=?, @EOID=?, @ClientID=?, @ArticleId=?, @ArticleSummary=?, @Opportunity=?, @Quote=?"
                                                    answer_values = (int(ap_id), int(topic_id), int(opportunity_id), int(client_id), int(langchain_best_article_id), str(article_summary), str(media_opportunities), str(quote))

                                                    cursor.execute(answer_insert_query, answer_values)
                                                    
                                                    output_apid_list = cursor.fetchone()
                                                    cursor.commit()

                                                    print(len(output_apid_list))
                                                    for row in output_apid_list:
                                                        ap_id = row
                                                else:
                                                    if sub_topic:
                                                        log_str = f"No best Articles found for topic {topic_name} and for sub topic {sub_topic} and client {client_name}"
                                                    else:
                                                        log_str = f"No best Articles found for topic {topic_name} and client {client_name}"
                                                    print(log_str)
                                                    writeLog(log_str)
                                        else:
                                            if sub_topic:
                                                log_str = f"No Opportunity found in database for topic {topic_name} and for sub topic {sub_topic} and client {client_name}"
                                            else:
                                                log_str = f"No Opportunity found in database for topic {topic_name} and client {client_name}"
                                            print(log_str)
                                            writeLog(log_str)
                                    else:
                                        if sub_topic:
                                            log_str = f"No best Articles found in langchain chains for topic {topic_name} and sub topic {sub_topic} and client {client_name}"
                                        else:
                                            log_str = f"No best Articles found in langchain chains for topic {topic_name} and client {client_name}"
                                        print(log_str)
                                        writeLog(log_str)

                                else:
                                    if sub_topic:
                                        log_str = f"All Articles were related to client {client_name} found for topic {topic_name} and sub topic {sub_topic}"
                                    else:
                                        log_str = f"All Articles were related to client {client_name} found for topic {topic_name}"
                                    print(log_str)
                                    writeLog(log_str)

                            else:
                                if sub_topic:
                                    log_str = f"No Relevant Articles found for topic {topic_name} and sub topic {sub_topic}"
                                else:
                                    log_str = f"No Relevant Articles found for topic {topic_name}"
                                print(log_str)
                                writeLog(log_str)
                        else:
                            if sub_topic:
                                log_str = f"No Articles found for topic {topic_name} and sub topic {sub_topic}"
                            else:
                                log_str = f"No Articles found for topic {topic_name}"
                            print(log_str)
                            writeLog(log_str)
                else:
                    log_str = f"No Clients found for topic {topic_name}"
                    print(log_str)
                    writeLog(log_str)
        except:
            print(traceback.print_exc())
            error_trace = traceback.format_exc()
            send_error(error_trace, topic_id, topic_name, emailids)



if __name__ == "__main__":
    main()

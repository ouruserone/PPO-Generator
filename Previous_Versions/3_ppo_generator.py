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



def get_similar_docs_id(vectorstore, topic_name):
    metadata_field_info=[
        AttributeInfo(
            name="id",
            description="The id of the news article", 
            type="string", 
        ),
        AttributeInfo(
            name="headline",
            description="The headline of the news article", 
            type="string or list[string]", 
        ),
        AttributeInfo(
            name="date",
            description="The date, news article was published", 
            type="integer", 
        ),
        AttributeInfo(
            name="publication",
            description="The name of the publication which published this news article", 
            type="string", 
        ),
        # AttributeInfo(
        #     name="domain",
        #     description="The domain of the news article",
        #     type="float"
        # ),
    ]
    document_content_description = "Brief summary of a news article"
    llm = OpenAI(temperature=0)
    # retriever = SelfQueryRetriever.from_llm(llm, vectorstore, document_content_description, metadata_field_info, verbose=True)

    retriever = SelfQueryRetriever.from_llm(
        llm, 
        vectorstore, 
        document_content_description, 
        metadata_field_info, 
        enable_limit=True,
        verbose=True
    )

    # This example only specifies a relevant query
    # input_text = "Top 5 Articles which are related to {}".format(topic_name)
    input_text = "Top 5 Articles which are about {}".format(topic_name)
    retrieved_docs_list = retriever.get_relevant_documents(input_text)
    # retrieved_docs_list = retriever.get_relevant_documents("Top 5 Articles which are related to sustainability and having domain Mobility")
    print(retrieved_docs_list)
    print(len(retrieved_docs_list))
    
    similar_id_list = []

    for retrieved_doc in retrieved_docs_list:
        metadata = retrieved_doc.metadata
        print(metadata)

        if 'id' in metadata:
            id = metadata['id']
            print('id:- ', id)
            similar_id_list.append(id)

    return retrieved_docs_list, similar_id_list


def get_relevant_articles(topic_id, topic, article_dict, openai_api_key, emailids):
    chat_gpt_error = False
    chat_gpt_error_str = ''
    relevant_article_list = []
    if article_dict:
        article_dict_keys_list = article_dict.keys()
        for article_dict_keys in article_dict_keys_list:
            article_id = article_dict_keys
            article = article_dict[article_dict_keys]
            try:
                URL = "https://api.openai.com/v1/chat/completions"

                api_key = openai_api_key

                messages = []

                ### Is this article about {topic}?\n \

                prompt = f'''
                You would be provided with article in the text delimited with triple backticks.\n\
                Can you please answer below question.\n \
                
                Is this article realted to {topic}?\n \
                
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

                    ChatGptwriteLog(str("ChatGpt for relevant articles Input for Article ID {} is:- \n {}\n").format(str(article_id), prompt))
                    ChatGptwriteLog(str("ChatGpt for relevant articles Response for Article ID {} is:- \n {}\n").format(str(article_id), output_response))

                    output_json = json.loads(output_response)
                    if str(output_json['Article Related']).lower() == 'yes':
                        relevant_article_list.append(article_id)
                else:
                    output_response = {}
                    chat_gpt_error = True
                    chat_gpt_error_str = ("There is Error in ChatGPT Response for relevant Article ID {}. Response Stauts code is {}").format(str(article_id), str(response.status_code))
                    send_error(chat_gpt_error_str, topic_id, topic, emailids)
            except:
                chat_gpt_error = True
                chat_gpt_error_str_trace = str(traceback.print_exc())
                chat_gpt_error_str = ("There is Error in ChatGPT Response for relevant Article ID {}.\n {}\n").format(str(article_id), str(chat_gpt_error_str_trace))
                send_error(chat_gpt_error_str, topic_id, topic, emailids)

    return relevant_article_list


def recheck_for_client(topic_id, topic_name, client, article_dict, openai_api_key, emailids):
    chat_gpt_error = False
    chat_gpt_error_str = ''
    article_for_client_list = []
    if article_dict:
        article_dict_keys_list = article_dict.keys()
        for article_dict_keys in article_dict_keys_list:
            article_id = article_dict_keys
            article = article_dict[article_dict_keys]
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

                    ChatGptwriteLog(str("ChatGpt for client recheck articles Input for Article ID {} is:- \n {}\n").format(str(article_id), prompt))
                    ChatGptwriteLog(str("ChatGpt for client recheck articles Response for Article ID {} is:- \n {}\n").format(str(article_id), output_response))

                    output_json = json.loads(output_response)
                    if str(output_json['Article about client']).lower() == 'no':
                        article_for_client_list.append(article_id)
                else:
                    output_response = {}
                    chat_gpt_error = True
                    chat_gpt_error_str = ("There is Error in ChatGPT Response for client recheck Article ID {}. Response Stauts code is {}").format(str(article_id), str(response.status_code))
                    send_error(chat_gpt_error_str, topic_id, topic_name, emailids)
            except:
                chat_gpt_error = True
                chat_gpt_error_str_trace = str(traceback.print_exc())
                chat_gpt_error_str = ("There is Error in ChatGPT Response for client recheck Article ID {}.\n {}\n").format(str(article_id), str(chat_gpt_error_str_trace))
                send_error(chat_gpt_error_str, topic_id, topic_name, emailids)

    return article_for_client_list


def client_media_opportunities(topic_id, topic_name, client, article_dict, opportunity_name, openai_api_key, emailids, client_desc):
    chat_gpt_error = False
    chat_gpt_error_str = ''
    best_article_id = None
    media_opportunities = None
    best_match_found = None
    article_summary = None
    article_prompt = 'You would be provided with 5 article with article id in triple backticks. \n \n'
    articles_str = ''
    if article_dict:
        article_dict_keys_list = article_dict.keys()
        for article_dict_keys in article_dict_keys_list:
            article_id = article_dict_keys
            article = article_dict[article_dict_keys]
            articles_str += f"```Article id: {article_id} \n Article text: {article}``` \n \n"
        try:
            URL = "https://api.openai.com/v1/chat/completions"

            api_key = openai_api_key

            messages = []

            question_prompt = f'''
            You would be provided with collection of articles with article id and article text.\n \n \
            This collection of articles will be provided with text delimited by triple backticks. \n 

            Your task is to help a communications team of {client}. \n 

            {client_desc} \n \n\
            
            Your task is to perform following actions. \n \n
            1 - Select best suited article for news jacketing exercise from the collection of articles provided with text delimited by triple backticks. \n \n
            2 - Give me a idea about {opportunity_name} for selected article in not more than 8 lines using line breaks and appropriate paragraphs. \n \n
            3 - Can you please provide output in JSON like:-\n \
            {{
                "Best match found": "yes or no",
                "id": "id of the news article",
                "article summary": "summary of the selected article",
                "media opportunity": "media opportunity"
            }}

            If there is no article suitable for the exercise of news jacketting of {client} then simply respond \""Best match found": "no"\"
            

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

                ChatGptwriteLog(str("ChatGpt for media opportunitiy {} Input for Topic {} is:- \n {}\n").format(str(opportunity_name), str(topic_name), prompt))
                ChatGptwriteLog(str("ChatGpt for media opportunitiy {} Response for Topic {} is:- \n {}\n").format(str(opportunity_name), str(topic_name), output_response))

                output_json = json.loads(output_response)
                if ('Best match found' in output_json):
                    if str(output_json['Best match found']).lower() == 'yes':
                        best_match_found = True
                    elif str(output_json['Best match found']).lower() == 'no':
                        best_match_found = False
                if ('id' in output_json) and ('media opportunity' in output_json) and ('article summary' in output_json):
                    best_article_id = str(output_json['id'])
                    media_opportunities = str(output_json['media opportunity'])
                    article_summary = str(output_json['article summary'])
            else:
                output_response = {}
                chat_gpt_error = True
                chat_gpt_error_str = ("There is Error in ChatGPT Response for media opportunities articles for Article ID {}. Response Stauts code is {}").format(str(article_id), str(response.status_code))
                send_error(chat_gpt_error_str, topic_id, topic_name, emailids)
        except:
            chat_gpt_error = True
            chat_gpt_error_str_trace = str(traceback.print_exc())
            chat_gpt_error_str = ("There is Error in ChatGPT Response for media opportunities articles for Article ID {}.\n {}\n").format(str(article_id), str(chat_gpt_error_str_trace))
            send_error(chat_gpt_error_str, topic_id, topic_name, emailids)

    return best_match_found, best_article_id, media_opportunities, article_summary

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

    index_name="ppo-data-test"
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

        try:
            if topic_name:
                retrieved_docs_list, similar_id_list = get_similar_docs_id(vectorstore, topic_name)
                print(similar_id_list)
                log_str = f"Pinecone retrieved Articles Id {str(similar_id_list)} found for topic {topic_name}"
                print(log_str)
                writeLog(log_str)

                article_dict = {}

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

                if article_dict:
                    relevant_article_list = get_relevant_articles(topic_id, topic_name, article_dict, openai_api_key, emailids)
                    log_str = f"Relevant retrieved Articles Id {str(relevant_article_list)} found for topic {topic_name}"
                    print(log_str)
                    writeLog(log_str)

                    if relevant_article_list:
                        get_client_query = "exec [USP_PPO_FetchApplicableClients] @TopicID=?"
                        values = (int(topic_id))
                        cursor.execute(get_client_query, values)
        
                        client_list = cursor.fetchall()
                        print(len(client_list))
                        
                        for row in client_list:
                            print(row[0])
                            client_id = row[0]
                            client_name = row[1]
                            client_desc = row[2]

                            ap_id = 0
                            relevant_article_dict = {}
                            for article_id in relevant_article_list:
                                relevant_article_dict[article_id] = article_dict[article_id]

                            article_for_client_list = recheck_for_client(topic_id, topic_name, client_name, relevant_article_dict, openai_api_key, emailids)
                            log_str = f"Client check retrieved Articles Id {str(article_for_client_list)} found for topic {topic_name}"
                            print(log_str)
                            writeLog(log_str)

                            if article_for_client_list:
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

                                        client_article_dict = {}
                                        for article_id in article_for_client_list:
                                            client_article_dict[article_id] = article_dict[article_id]

                                        best_match_found, best_article_id, media_opportunities, article_summary = client_media_opportunities(topic_id, topic_name, client_name, client_article_dict, opportunity_name, openai_api_key, emailids, client_desc)
                                        log_str = f"Best retrieved Articles Id {str(best_article_id)} found for topic {topic_name}"
                                        print(log_str)
                                        writeLog(log_str)

                                        if best_match_found:
                                            answer_insert_query = "exec [USP_AnswerPPO_Insert] @APID=?, @TopicID=?, @EOID=?, @ClientID=?, @ArticleId=?, @ArticleSummary=?, @Opportunity=?"
                                            answer_values = (int(ap_id), int(topic_id), int(opportunity_id), int(client_id), int(best_article_id), str(article_summary), str(media_opportunities))

                                            cursor.execute(answer_insert_query, answer_values)
                                            
                                            output_apid_list = cursor.fetchone()
                                            cursor.commit()

                                            print(len(output_apid_list))
                                            for row in output_apid_list:
                                                ap_id = row
                                        else:
                                            log_str = f"No best Articles found for topic {topic_name}"
                                            print(log_str)
                                            writeLog(log_str)
                                else:
                                    log_str = f"No Opportunity found in database for topic {topic_name}"
                                    print(log_str)
                                    writeLog(log_str)
                            else:
                                log_str = f"All Articles were related to clients found for topic {topic_name}"
                                print(log_str)
                                writeLog(log_str)
                    else:
                        log_str = f"No Relevant Articles found for topic {topic_name}"
                        print(log_str)
                        writeLog(log_str)
                else:
                    log_str = f"No Articles found for topic {topic_name}"
                    print(log_str)
                    writeLog(log_str)
        except:
            print(traceback.print_exc())
            error_trace = traceback.format_exc()
            send_error(error_trace, topic_id, topic_name, emailids)



if __name__ == "__main__":
    main()

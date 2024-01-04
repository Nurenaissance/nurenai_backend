! pip install numpy
! pip install openai
! pip install pymongo
! pip install python-dotenv
! pip install azure-core
! pip install azure-cosmos
! pip install tenacity  

import json
import datetime
import time

from azure.core.exceptions import AzureError
from azure.core.credentials import AzureKeyCredential
import pymongo

import openai
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt

from dotenv import dotenv_values

# specify the name of the .env file name 
env_name = "example.env" # following example.env template change to your own .env file name
config = dotenv_values(env_name)

cosmosdb_endpoint = config['cosmos_db_api_endpoint']
cosmosdb_key = config['cosmos_db_api_key']
cosmosdb_connection_str = config['cosmos_db_connection_string']

COSMOS_MONGO_USER = config['cosmos_db_mongo_user']
COSMOS_MONGO_PWD = config['cosmos_db_mongo_pwd']
COSMOS_MONGO_SERVER = config['cosmos_db_mongo_server']

openai.api_type = config['openai_api_type']
openai.api_key = config['openai_api_key']
openai.api_base = config['openai_api_endpoint']
openai.api_version = config['openai_api_version']
embeddings_deployment = config['openai_embeddings_deployment']
completions_deployment = config['openai_completions_deployment']

# OR Load text-sample_w_embeddings.json which has embeddings pre-computed
data_file = open(file="../../DataSet/AzureServices/text-sample_w_embeddings.json", mode="r") 
data = json.load(data_file)
data_file.close()

# Take a peek at one data item
print(json.dumps(data[0], indent=2))


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def generate_embeddings(text):
    '''
    Generate embeddings from string of text.
    This will be used to vectorize data and user input for interactions with Azure OpenAI.
    '''
    response = openai.Embedding.create(
        input=text, engine="text-embedding-ada-002")
    embeddings = response['data'][0]['embedding']
    time.sleep(0.5) # rest period to avoid rate limiting on AOAI for free tier
    return embeddings


# Generate embeddings for title and content fields
n = 0
for item in data:
    n+=1
    title = item['title']
    content = item['content']
    title_embeddings = generate_embeddings(title)
    content_embeddings = generate_embeddings(content)
    item['titleVector'] = title_embeddings
    item['contentVector'] = content_embeddings
    item['@search.action'] = 'upload'
    print("Creating embeddings for item:", n, "/" ,len(data), end='\r')
# Save embeddings to sample_text_w_embeddings.json file
with open("../../DataSet/AzureServices/text-sample_w_embeddings.json", "w") as f:
    json.dump(data, f)


mongo_conn = "mongodb+srv://"+COSMOS_MONGO_USER+":"+COSMOS_MONGO_PWD+"@"+COSMOS_MONGO_SERVER+"?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000"
mongo_client = pymongo.MongoClient(mongo_conn)

# create a database called TutorialDB
db = mongo_client['ExampleDB']

# Create collection if it doesn't exist
COLLECTION_NAME = "ExampleCollection"

collection = db[COLLECTION_NAME]

if COLLECTION_NAME not in db.list_collection_names():
    # Creates a unsharded collection that uses the DBs shared throughput
    db.create_collection(COLLECTION_NAME)
    print("Created collection '{}'.\n".format(COLLECTION_NAME))
else:
    print("Using collection: '{}'.\n".format(COLLECTION_NAME))
## Use only if re-reunning code and want to reset db and collection
collection.drop_index("VectorSearchIndex")
mongo_client.drop_database("ExampleDB")

db.command({
  'createIndexes': 'ExampleCollection',
  'indexes': [
    {
      'name': 'VectorSearchIndex',
      'key': {
        "contentVector": "cosmosSearch"
      },
      'cosmosSearchOptions': {
        'kind': 'vector-ivf',
        'numLists': 1,
        'similarity': 'COS',
        'dimensions': 1536
      }
    }
  ]
})


db.command(
{ 
    "createIndexes": "ExampleCollection",
    "indexes": [
        {
            "name": "VectorSearchIndex",
            "key": {
                "contentVector": "cosmosSearch"
            },
            "cosmosSearchOptions": { 
                "kind": "vector-hnsw", 
                "m": 16, # default value 
                "efConstruction": 64, # default value 
                "similarity": "COS", 
                "dimensions": 1536
            } 
        } 
    ] 
}
)


collection.insert_many(data)

# Simple function to assist with vector search
def vector_search(query, num_results=3):
    query_embedding = generate_embeddings(query)
    embeddings_list = []
    pipeline = [
        {
            '$search': {
                "cosmosSearch": {
                    "vector": query_embedding,
                    "path": "contentVector",
                    "k": num_results #, "efsearch": 40 # optional for HNSW only 
                },
                "returnStoredSource": True }},
        {'$project': { 'similarityScore': { '$meta': 'searchScore' }, 'document' : '$$ROOT' } }
    ]
    results = collection.aggregate(pipeline)
    return results


query = "What are the services for running ML models?"
results = vector_search(query)
for result in results: 
#     print(result)
    print(f"Similarity Score: {result['similarityScore']}")  
    print(f"Title: {result['document']['title']}")  
    print(f"Content: {result['document']['content']}")  
    print(f"Category: {result['document']['category']}\n")  
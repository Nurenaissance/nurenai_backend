import os

settings = {
    'host': os.environ.get('ACCOUNT_HOST', 'https://storagecosmos1.documents.azure.com:443/'),
    'master_key': os.environ.get('ACCOUNT_KEY', 'oo2YFN36irdNUBLyaZSkgZJw9ekkwXhp6iM0iHjUb4lgV7hOxXhbSmMd2lWuBLHteybtLa5NKSY2ACDbcHorkQ=='),
    'database_id': os.environ.get('COSMOS_DATABASE', 'ToDoList'),
    'container_id': os.environ.get('COSMOS_CONTAINER', 'Items'),
}
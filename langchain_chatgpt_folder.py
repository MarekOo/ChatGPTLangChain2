import logging
import os
import sys
import shutil

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
import hashlib


def hash_directory_content(data_dir):
    file_hashes = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as f:
                file_content = f.read()
                hash_content = file_content
                # calculate the filesize of all files
                filesize = os.path.getsize(file_path)
                hash_content += str(filesize).encode()
                file_hash = hashlib.sha256(hash_content).hexdigest()
                file_hashes.append(file_hash)

    directory_hash = hashlib.sha256(''.join(file_hashes).encode()).hexdigest()
    return directory_hash


class LangChain:
    def __init__(self, model="gpt-3.5-turbo", data_dir="data/", persist_dir="persist", persist=True):
        self.model = model
        self.data_dir = data_dir
        self.persist = persist
        self.persist_dir = persist_dir
        self.index = None
        self.preprocess()

    def preprocess(self):
        logging.info(f"Initializing with model: {self.model}, data_dir: {self.data_dir}, persist: {self.persist}, persist_dir: {self.persist_dir}")
        # Access the environment variable
        load_dotenv()
        api_key = os.getenv("API_KEY")
        os.environ["OPENAI_API_KEY"] = api_key

        # Enable persistence, save & reuse of index data
        # create a hash for content of the data directory
        # this is used to check if the data has changed since the last run
        # if it has, the index is recreated
        # if not, the index is loaded from disk
        # this is only used if persist is True

        # Create the hash for the data directory
        data_hash = hash_directory_content(self.data_dir)

        logging.info("Data hash: " + data_hash)
        # if the persist directory exists, check if the data has changed
        # if it has, delete the persist directory
        if self.persist and os.path.exists(self.persist_dir):
            if not os.path.exists(self.persist_dir + "/data_hash"):
                # delete directory persist
                shutil.rmtree(self.persist_dir)
                logging.info("Persistent directory deleted, data has changed...\n")
            else:
                with open(self.persist_dir + "/data_hash", "r") as f:
                    if f.read() != data_hash:
                        logging.info("Data has changed, deleting index...\n")
                        shutil.rmtree(self.persist_dir)

        if self.persist and os.path.exists(self.persist_dir):
            logging.info("Reusing the old index...\n")
            vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=OpenAIEmbeddings())
            self.index = VectorStoreIndexWrapper(vectorstore=vectorstore)
        else:
            logging.info("Creating new index...\n")
            loader = DirectoryLoader(self.data_dir)
            if self.persist:
                logging.info("Saving index to folder: " + self.persist_dir + " for later use\n")
                self.index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": self.persist_dir}).from_loaders([loader])
                # save hash for data directory in persist directory
                with open(self.persist_dir + "/data_hash", "w") as f:
                    f.write(data_hash)
            else:
                self.index = VectorstoreIndexCreator().from_loaders([loader])

    def do_prompt(self, chat_history=[], query=None):
        chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model=self.model),
            retriever=self.index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        )
        result = chain({"question": query, "chat_history": chat_history})
        return result


# change logging to print info, uncomment of you want to see the info messages
logging.basicConfig(level=logging.INFO)

# create chat_history array to store prompts and answers
chat_history = []

# create langchain object
myLangChain = LangChain()
query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

while True:
    if not query:
        query = input("Prompt (type 'q' to quit): ")
    if query in ['quit', 'q']:
        sys.exit()
    query_result = myLangChain.do_prompt(chat_history=chat_history, query=query)
    print(query_result['answer'])
    chat_history.append((query, query_result['answer']))
    query = None

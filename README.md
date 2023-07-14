# ChatGPTLangChain 2
Little demo on how to use ChatGPT on your own data using LangChain.

## Use
Create a .env file in the root folder with your openai api key as API_KEY=yourkeyhere

Just copy all your data (.txt, .pdf) into the "data" folder and run the "langchain_chatgpt_folder.py" in the command line. The script will create an index and save it the folder "persist" for later use it, so it won't have to be created each time. Use the prompt to chat with your data.

## Background
The script will create an index and save it in "persist". By default if the data does not change the saved index will be used to save traffic and transaction costs. The changes are recognized by a hash that is calculated on the data folder content (file names and size)

# RAG on Bedrock

This demo is available to run on [Streamlit Cloud](https://aws-summit-sydney-2024-demo.streamlit.app), or you can run the demo yourself using the instrcutions below

Demo Features:
- Allows a user top upload content using either:
    - File Upload
    - Website URL
    - Raw Text
- Keeps track of each user's Vector data and Conversation History separately. That way one users content does not interfere with any other user.
- Allows the user to choose which Bedrock Foundation Model to use.


### Setup to run your own instance

1. Create Streamlit Secrets
The following attributes must be provided in your secrets.toml.
[See Streamlit docs fro more help](https://docs.streamlit.io/develop/concepts/connections/secrets-management)

    - ASTRA_VECTOR_ENDPOINT
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_DEFAULT_REGION

    If you want to use LangSmith, then these are also required:

    - LANGCHAIN_ENDPOINT
    - LANGCHAIN_API_KEY

2. Run the app

    `streamlit run app_bedrock.py`



## Using the demo

1. Provide content

    - Use the sidebar to provide content that will be referenced in the RAG workflow.
    - You can choose from:
        - Uploading a file
        - Providing a URL 
        - Providing text

2. Select your preferred LLM from the available Bedrock Foundation Models

3. Ask a question
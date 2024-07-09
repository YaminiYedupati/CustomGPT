# CustomGPT
This a custom Q&A based answering model that is trained on SonarQube documentation.
The purpose of this LLM is to provide AI like support for users trying to find answers to their questions related to SonarQube.

Tech stack used:
- Python
- LangChain
- HuggingFace
- StreamLit (LLM integrated with StreamLit's API to serve as UI)

To use the model:
1. Clone the repository.
2. Run the command streamlit run .\ExtractWebData.py in your favorite editor. Note: All module dependencies are currently installed manually. WIP to automatically install the requirements for the usecase.
3. Ask a question related to SonarQube in the dialog box that pops up in your browser!

Example:

![image](https://github.com/YaminiYedupati/CustomGPT/assets/147988230/8884bde4-e054-477c-b96c-45c160e7662a)

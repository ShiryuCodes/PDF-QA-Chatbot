# 📚 PDF QA Chatbot with Gemini

An intelligent PDF question-answering chatbot built with **Streamlit**, **LangChain**, and **Google Gemini**. Upload any PDF document and ask questions about its content using advanced RAG (Retrieval Augmented Generation) technology.

## 🚀 Live Demo

🔗 **[Try it live on Hugging Face Spaces]()**

## ✨ Features

- **Easy PDF Upload**: Drag and drop any PDF document
- **AI-Powered QA**: Powered by Google Gemini 2.5 Flash for intelligent responses
- **Interactive Chat**: Clean chat interface with conversation history
- **Smart Search**: Uses vector similarity search for relevant context
- **Fast Processing**: Efficient document chunking and embedding
- **Accurate Answers**: RAG technology ensures responses are grounded in your document
- **Privacy First**: Your API key and documents are never stored

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.5 Flash
- **Framework**: LangChain
- **Vector Store**: ChromaDB  
- **Embeddings**: Google Generative AI Embeddings
- **PDF Processing**: PyPDF

## Prerequisites

- Python 3.8+
- Google Gemini API Key ([Get it free here](https://makersuite.google.com/app/apikey))

## 🔧 Installation

1. **Clone the repository**
```bash
git clone https://github.com/ShiryuCodes/PDF-QA-Chatbot.git
cd PDF-QA-Chatbot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

## 📖 Usage

1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to get your free Gemini API key
2. **Enter API Key**: Paste your API key in the sidebar (it's secure and not stored)
3. **Upload PDF**: Choose any PDF document you want to analyze
4. **Process Document**: Click "Process PDF" to prepare the document for questions
5. **Ask Questions**: Start asking questions about your PDF content!
6. **Chat History**: View all your previous questions and answers in the session

## How It Works

This application uses **RAG (Retrieval Augmented Generation)** technology:

1. **Document Processing**: PDF is split into manageable chunks
2. **Vector Embeddings**: Each chunk is converted to vector embeddings using Google's embedding model
3. **Vector Storage**: Embeddings are stored in ChromaDB for fast similarity search
4. **Query Processing**: Your questions are also converted to embeddings
5. **Retrieval**: Most relevant document chunks are retrieved based on similarity
6. **Generation**: Gemini generates accurate answers using the retrieved context

## 📁 Project Structure

```
PDF-QA-Chatbot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── packages.txt          # System dependencies (for Hugging Face)
└── Notebook/
    └── chatbot_for_pdfs.ipynb  # Original development notebook
```

## 🔒 Privacy & Security

- ✅ **API keys are never stored** - entered fresh each session
- ✅ **PDFs are processed temporarily** - no permanent storage
- ✅ **Local processing** - your documents don't leave the session
- ✅ **No tracking** - we don't collect any user data

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Need Help?

If you encounter any issues or have questions:

1. **Check the [Issues](https://github.com/ShiryuCodes/PDF-QA-Chatbot/issues)** page
2. **Create a new issue** with detailed description
3. **View the original notebook** for implementation details: [chatbot_for_pdfs.ipynb](https://github.com/ShiryuCodes/PDF-QA-Chatbot/blob/main/Notebook/chatbot_for_pdfs.ipynb)

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

---

**Built with ❤️ by Shivang**

*From notebook prototype to production app - [development notebook](https://github.com/ShiryuCodes/PDF-QA-Chatbot/blob/main/Notebook/chatbot_for_pdfs.ipynb)!*

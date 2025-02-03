# ATS - Resume Analyzer
ATS - Resume Analyzer is a Streamlit-based application that analyzes resume to identify strengths and areas for improvement, compares them with job descriptions to suggest relevant skills and, calculates an ATS score

## Technologies Used
- Python 
- Streamlit (UI Development) 
- OpenAI API (GPT-3.5 Turbo for resume analysis) 
- Hugging Face Embeddings (Text vectorization) 
- FAISS (Facebook AI Similarity Search) (Efficient resume-job matching) 
- PyPDF2 (Extracting text from resumes) 
- LangChain (RAG-based pipeline for LLM interaction) 

## Installation
1.  Clone the Repository
``` bash
git clone https://github.com/SanjanaRedd26/Resume-Analyzer.git
cd Resume-Analyzer
 ```
2. Set Up a Virtual Environment (Optional but Recommended)
``` bash
python -m venv venv
source venv/bin/activate
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Set Up API Keys
- Create a .env file in the src directory and add your OpenAI API key:
``` bash
OPENAI_API_KEY=your_openai_api_key_here
```
5. To Run the app
``` bash
streamlit run src/main.py
```
## Contributions

Open to contributions! Feel free to fork the repo and submit a pull request with improvements or new features.







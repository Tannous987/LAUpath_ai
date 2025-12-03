# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Set Up API Key
1. Get your Gemini API key from: https://makersuite.google.com/app/apikey
2. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_key_here
   ```

### Step 3: Set Up RAG System
```bash
python setup_rag.py
```
Wait for the script to process all PDFs (this may take 2-5 minutes).

### Step 4: Run the Application
```bash
streamlit run app.py
```

### Step 5: Start Using LAUpath AI!
1. Open the sidebar and enter your academic information
2. Ask questions like:
   - "What are the admission requirements?"
   - "Analyze my profile"
   - "Help me choose a major"
   - "Tell me about Computer Science"

## ✅ Verification Checklist

- [ ] Dependencies installed
- [ ] `.env` file created with API key
- [ ] `setup_rag.py` completed successfully
- [ ] `vector_db/` directory created
- [ ] Streamlit app runs without errors
- [ ] Can chat with the agent
- [ ] Tools are being called (visible in UI)

## 🐛 Troubleshooting

### "GEMINI_API_KEY not found"
- Make sure `.env` file exists in project root
- Check that the key is set correctly: `GEMINI_API_KEY=your_key`

### "Vector database not found"
- Run `python setup_rag.py` first
- Check that `vector_db/` directory was created

### "No PDF files found"
- Ensure PDF files are in `data/pdfs/` directory
- Check file names end with `.pdf`

### Import errors
- Make sure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --upgrade`

## 📚 Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for architecture details
3. Explore the code to understand how it works
4. Customize tools and prompts for your needs


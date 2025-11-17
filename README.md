# 🤖 LLM-as-Judge Evaluation Framework (Groq)

A fully interactive Streamlit app that uses a **Groq-hosted LLM (Llama 3 models)** to act as an **impartial judge** for evaluating and comparing two model outputs.  
This tool helps you measure subjective qualities—such as *creativity*, *brand tone adherence*, *code readability*, or any custom metric you define.

Perfect for:
- A/B testing model responses  
- Benchmarking prompt variations  
- Building custom evaluation frameworks  
- Automating subjective model scoring  

---

## 🚀 Features

### 🔍 Compare Model Outputs
Paste two outputs (A and B), describe the task, and let the LLM judge evaluate them.

### 🎯 Multiple Evaluation Criteria
Choose from built-in presets:
- Creativity
- Brand Tone Adherence
- Code Readability

Or define **your own custom criterion** (e.g., "Factual Accuracy", "Politeness", "Conciseness", etc.).

### 📊 Structured Evaluation Output
Every evaluation returns:
- Score for Output A
- Score for Output B
- Judge's explanation for each
- A final winner (A, B, or tie)
- A combined overall comment

### 🧠 Powered by Groq
Uses the Groq API for ultra-fast inference with **Llama 3** models.

### 🖥️ Clean UI
Built entirely with Streamlit — responsive and easy to use.

---

## 🗂️ Project Structure
```text
llm-as-judge-groq/
├─ app.py # Streamlit web UI
├─ judge.py # Groq evaluation logic
├─ prompts.py # Prompt templates and criteria presets
├─ config.py # Loads environment variables
├─ requirements.txt # Python dependencies
├─ .env.example # Example env file (no real keys)
├─ .gitignore # Ignore secrets and build files
└─ README.md # Project documentation
```

## 🧪 Usage Example

1. Describe the task:
"Write a friendly tweet announcing our new AI tool."
2. Paste two different outputs from any models.
3. Select a criterion (e.g., Creativity)
4. Click Evaluate Outputs

You’ll receive:

- A score for Output A
- A score for Output B
- Explanations
- A final winner
- Overall comments

## 🔒 Security Notes

- Never commit your .env file.
- Your .gitignore already ignores .env.
- If a key is ever exposed, regenerate it immediately in the Groq console.

## 🛠️ Future Improvements (ideas)

- 📝 Multi-criteria scoring (e.g., Creativity + Accuracy + Tone)
- 📥 Upload CSV for batch evaluation
- 📤 Export results (CSV/JSON)
- 🧪 Integrate with experiment tracking (Weights & Biases, LangSmith, etc.)
- 🔌 Plug in more models (OpenAI, Anthropic, HuggingFace, etc.)

## ⭐ Contributing

- Pull requests are welcome!
- Feel free to submit issues or feature requests.

## 📄 License

- MIT License — free to use, modify, and distribute.

Enjoy using the LLM-as-Judge Evaluation Framework! If you build something cool on top of this, share it! 🚀

## Author

Created by Syed Waleed Ahmed
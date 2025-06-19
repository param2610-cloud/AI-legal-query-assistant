# Quick Start Guide - AI Legal Assistant

## 🚀 Get Started in 3 Steps

### Step 1: Run Setup
```bash
./setup.sh
```

### Step 2: Start the Web Interface
```bash
streamlit run streamlit_app.py
```

### Step 3: Ask Legal Questions!
Open http://localhost:8501 and start asking questions like:
- "What are my consumer rights?"
- "How do I file a complaint for defective products?"
- "What are child labor laws in India?"

## 🛠️ Alternative: CLI Interface

```bash
# Check system requirements
python cli.py --check

# Run interactive CLI
python cli.py
```

## 📱 Web Interface Features

- **💬 Chat Interface**: Natural conversation with AI
- **📚 Legal Acts**: Access to multiple Indian laws
- **🔍 Smart Search**: Find relevant legal provisions
- **📋 Quick Actions**: Common legal questions
- **⚠️ Safety**: Built-in disclaimers and warnings

## 🧠 How It Works

1. **Your Question**: Ask in plain English
2. **AI Analysis**: System searches legal documents
3. **Smart Response**: Get simplified explanations
4. **Related Laws**: See relevant legal provisions
5. **Next Steps**: Practical guidance provided

## 💡 Example Interactions

### Consumer Rights
**You**: "I bought a defective phone online. What can I do?"

**AI**: "Under the Consumer Protection Act, you have several rights:
- Right to replacement or refund for defective products
- Right to compensation for mental agony
- 30-day return policy for online purchases
- Can file complaint with District Consumer Forum..."

### Employment Issues
**You**: "My boss is making me work 14 hours daily. Is this legal?"

**AI**: "Under Indian labor laws:
- Maximum working hours: 8 hours per day, 48 hours per week
- Overtime must be paid at double rate
- Weekly rest day is mandatory
- You can file complaint with Labor Commissioner..."

## ⚠️ Important Notes

- **Educational Use Only**: Not a substitute for legal advice
- **Consult Lawyers**: For complex matters, always consult professionals
- **Stay Updated**: Laws may change, verify current regulations
- **Local Variations**: Some laws may vary by state

## 🔧 Troubleshooting

### Common Issues:

1. **"Ollama not found"**
   - Install: `curl -fsSL https://ollama.ai/install.sh | sh`
   - Start: `ollama serve`

2. **"Import errors"**
   - Run: `pip install -r requirements.txt`

3. **"Model not available"**
   - Download: `ollama pull llama3.2:3b`

4. **"System slow"**
   - Ensure 8GB+ RAM available
   - Close other applications

### Get Help:
- Run: `python cli.py --check` to diagnose issues
- Run: `python cli.py --setup` for setup instructions
- Check logs in `logs/` directory

## 🎯 Pro Tips

1. **Be Specific**: Describe your situation clearly
2. **Ask Follow-ups**: Continue the conversation for clarity
3. **Check Sources**: Review the legal acts mentioned
4. **Save Important Info**: Copy important responses
5. **Verify**: Always verify with official sources

---

**🏛️ Making Indian Laws Accessible to Everyone!**

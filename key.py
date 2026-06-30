

from groq import Groq
client = Groq(api_key="GROQ_API_KEY")
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "say hello"}],
    model="llama-3.3-70b-versatile",
)
print(response.choices[0].message.content)
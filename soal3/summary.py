import pandas as pd
import argparse
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import os

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_gemini_content(feedback_text, num_words=5):
    prompt=f'''You are Customer Feedback summarizer. You will be taking the feedback text
                and summarizing the entire text and providing the important summary
                within {num_words} words. Please provide the 5 words summary of the text given here: '''

    model=genai.GenerativeModel("gemini-2.0-flash-lite")
    response=model.generate_content(prompt+feedback_text)
    return response.text

def main():
    parser = argparse.ArgumentParser(description="Summarize customer feedback using Gemini.")
    parser.add_argument("file_name", help="Name of the JSON file in the current directory.")
    parser.add_argument("--column", default="feedback", help="Name of the feedback column. Default: 'feedback'")
    parser.add_argument("--num_words", type=int, default=5, help="Number of words for summarization. Default: 5")
    parser.add_argument("--output", default="summarized_data.csv", help="Output CSV file. Default: summarized_data.csv")

    args = parser.parse_args()

    file_path = os.path.join(os.getcwd(), args.file_name)

    try:
        df = pd.read_json(file_path, lines=True)
    except Exception as e:
        print("Error reading JSON:", e)
        return

    if args.column not in df.columns:
        print(f"Column '{args.column}' not found in JSON.")
        return

    df['summary'] = df[args.column].apply(lambda x: generate_gemini_content(x, args.num_words))
    df.to_csv(args.output, index=False)
    print(f"Summarization complete! Saved as {args.output}")

if __name__ == "__main__":
    main()
# Camera Dashboard App v2 with Chatbot

This Streamlit app combines:
- dashboard pages for review, sentiment, topic, and trend analysis
- a chatbot page backed by the built knowledge base in `kb_output`

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data files

Keep the main data file in one of these places:
- `data/camera_sentiment_label.xlsx` inside the app folder
- or another project path already supported by `app.py`

## Chatbot requirements

The chatbot page expects a built knowledge base folder named `kb_output` in one of these places:
- next to `app.py`
- one level above the app folder
- current working directory

Build it first with your embedding pipeline, for example:

```bash
python build_kb_and_index.py --input_xlsx camera_sentiment_label.xlsx --output_dir kb_output
```

## API key

For final answer generation on the chatbot page, provide a BigModel API key either:
- in the page input box
- or as environment variable `BIGMODEL_API_KEY`

Without an API key, the chatbot page can still retrieve and display evidence.

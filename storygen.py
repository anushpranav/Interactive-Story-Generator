from flask import Flask, render_template, request, jsonify
from transformers import pipeline, GPT2LMHeadModel, AutoTokenizer
import spacy

app = Flask(__name__)

# Initialize the model and tokenizer
model_name = "aspis/gpt2-genre-story-generation"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to save the generated story to a file
def save_story_to_file(story_text):
    filename = "generated_story.txt"
    with open(filename, "w") as file:
        file.write(story_text)
    return f"Story saved to '{filename}'."

# Function to extract named entities from the generated story
def extract_named_entities(story_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(story_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Function for relation extraction
def extract_relations(story_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(story_text)

    relations = []
    for token in doc:
        if token.dep_ in ["attr", "agent", "pobj"]:  # Modify this condition as needed
            relations.append((token.head.text, token.dep_, token.text))

    return relations

# Function for chunking
def extract_chunks(story_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(story_text)

    noun_chunks = [chunk.text for chunk in doc.noun_chunks]
    return noun_chunks

# Function to generate the story based on the input prompt
def generate_story(input_prompt, max_length=300):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    story = generator(input_prompt, max_length=max_length, do_sample=True)
    generated_text = story[0]['generated_text']
    return generated_text

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for story generation
@app.route('/generate', methods=['POST'])
def story_generation():
    data = request.get_json()
    input_prompt = data['inputPrompt']
    max_length = int(data.get('maxLength', 300))

    # Generate story and additional features
    generated_story = generate_story(input_prompt, max_length)
    save_result = save_story_to_file(generated_story)
    named_entities = extract_named_entities(generated_story)
    relations = extract_relations(generated_story)
    noun_chunks = extract_chunks(generated_story)

    return jsonify({
        'generatedText': generated_story,
        'saveResult': save_result,
        'namedEntities': named_entities,
        'relations': relations,
        'nounChunks': noun_chunks
    })

if __name__ == '__main__':
    app.run(debug=True)

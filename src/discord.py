from dotenv import load_dotenv
import os
import json
import discord
from discord import app_commands
from discord.ext import commands
from main import song_cover_pipeline  # Assuming this function generates the song

intents = discord.Intents.default()
intents.messages = True
bot = commands.Bot(command_prefix="!", intents=intents)
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

# Load public models
with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
    public_models = json.load(infile)

def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user}')
    try:
        synced = await bot.tree.sync()
        print(f"Synced {len(synced)} slash commands")
    except Exception as e:
        print(e)

# Command to list available models
@bot.tree.command(name="list_models", description="List all available RVC models.")
async def list_models(interaction: discord.Interaction):
    voice_models = get_current_models(rvc_models_dir)
    if voice_models:
        models_str = "\n".join(voice_models)
        await interaction.response.send_message(f"**Available Models:**\n{models_str}")
    else:
        await interaction.response.send_message("No models found.")

# Command to generate AI cover
@bot.tree.command(name="generate_cover", description="Generate an AI cover song.")
@app_commands.describe(
    model="Choose a voice model.",
    song_input="YouTube link or local file path.",
    pitch="Pitch adjustment (-3 to 3).",
    output_format="Output file format (mp3 or wav)."
)
async def generate_cover(
    interaction: discord.Interaction,
    model: str,
    song_input: str,
    pitch: int = 0,
    output_format: str = "mp3"
):
    if model not in get_current_models(rvc_models_dir):
        await interaction.response.send_message(f"Model `{model}` not found!", ephemeral=True)
        return

    # Notify user that processing has started
    await interaction.response.send_message(f"Generating AI cover with model `{model}`. This may take a while...")

    # Run song generation pipeline
    try:
        output_file = song_cover_pipeline(
            song_input=song_input,
            rvc_model=model,
            pitch=pitch,
            output_format=output_format,
            output_dir=output_dir,
        )
        # Send the result
        await interaction.followup.send(file=discord.File(output_file, filename=os.path.basename(output_file)))
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}")

# Command to upload a model
@bot.tree.command(name="upload_model", description="Upload a new voice model.")
@app_commands.describe(model_name="Name your model.", zip_file="Upload the model zip file.")
async def upload_model(interaction: discord.Interaction, model_name: str, zip_file: discord.Attachment):
    try:
        zip_path = os.path.join(BASE_DIR, f"{model_name}.zip")
        await zip_file.save(zip_path)

        # Extract and move files to the model directory
        extraction_folder = os.path.join(rvc_models_dir, model_name)
        if os.path.exists(extraction_folder):
            await interaction.response.send_message(f"Model `{model_name}` already exists!", ephemeral=True)
            return

        extract_zip(extraction_folder, zip_path)
        await interaction.response.send_message(f"Model `{model_name}` uploaded successfully!")
    except Exception as e:
        await interaction.response.send_message(f"An error occurred: {e}")

# Helper function to extract zip files
def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder, exist_ok=True)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    # Handle file organization
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            file_path = os.path.join(root, name)
            if name.endswith('.pth') or name.endswith('.index'):
                shutil.move(file_path, extraction_folder)

if __name__ == "__main__":
    api_key = os.getenv("YOUR_DISCORD_BOT_TOKEN")
    bot.run(api_key)

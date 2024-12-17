import runpod
from rp_schemas import INPUT_SCHEMA
from runpod.serverless.utils.rp_validator import validate
import requests, base64
from inference_corlorizer import inference_colorizer
import io

def prepare_image_input(job):
    """
    Prepares image data for processing from various input types.

    Args:
        job:
        - source (str): The image source (can be a URL, file path, or base64 string).
        - input_type (str): The type of input provided (options: 'url', 'file', 'base64').

    Returns:
    - dict: A job dictionary with the image encoded in base64.
    """
    source = job.get('source')
    input_type = job.get('input_type')
    image_data = None

    if input_type == 'url':
        # Load image from a URL
        response = requests.get(source)
        image_data = response.content

    elif input_type == 'file':
        # Load image from a local file path
        with open(source, 'rb') as image_file:
            image_data = image_file.read()

    elif input_type == 'base64':
        # The source is already a base64 string
        image_data = base64.b64decode(source)

    elif input_type == 'upload':

        image_data = source.read()
    # Convert the image data to base64 (for uniformity in job input)
    # image_base64 = base64.b64encode(image_data).decode('utf-8')
    return image_data


def main(job):
    job_input = job['input']
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    image_data = prepare_image_input(job_input)
    result = inference_colorizer(img_path=io.BytesIO(image_data), use_gpu=True)
    return {
        'image':base64.b64encode(image_data).decode('utf-8'),
        'result_base64': result,
        'message':'Image processed successfully',
    }

runpod.serverless.start({"handler":main})

# url = "/home/minhthuy/Desktop/GFPGAN/inputs/whole_imgs/ImportedPhoto_1732011021324.jpg"

# def convert_image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         base64_string = base64.b64encode(image_file.read()).decode('utf-8')
#     return base64_string
# image_base64 = convert_image_to_base64(url)
# job = {
#     "input":{
#         "source":image_base64,
#         "input_type":"base64",
#     }
# }
# main(job)

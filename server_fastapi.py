import base64
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import vgg16_model

app = FastAPI()

# Load the model once
try:
    vgg16 = vgg16_model.load_model()
except Exception as e:
    raise Exception(f"failed to load model: {e}")

# cors 이슈
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_api():
    return "This is the Jellyfish API"

@app.post("/")
async def post_image(
    image: UploadFile = File(..., description="upload an image file")
):
    try:
        if not image.filename:
            raise HTTPException(status_code=400, detail="no image file")
            
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="image file only")
            
        image_bytes = await image.read()
        result = await vgg16_model.prediction_model(vgg16, image_bytes)
        return result
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to predict: {str(e)}")

# Run the server
if __name__ == "__main__":
    uvicorn.run("server_fastapi:app",
            reload= True,   # Reload the server when code changes
            host="127.0.0.1",   # Listen on localhost 
            port=5000,   # Listen on port 5000 
            log_level="info"   # Log level
            )
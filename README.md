# AITest2
App for detection of images and video generated with AI
Part1 Xvgtase88!


backend/
 ├── main.py
 ├── ai_detector.pth
 └── requirements.txt


fastapi
uvicorn
torch
torchvision
opencv-python
pillow
python-multipart

from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as T
import cv2
import tempfile
import os

app = FastAPI()

# Load model
model = torch.load("ai_detector.pth", map_location="cpu")
model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

def predict_pil(img: Image.Image) -> float:
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(img)).item()
    return prob

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    prob = predict_pil(img)

    return {
        "ai_probability": round(prob, 3),
        "label": "AI-generated" if prob > 0.6 else "Likely real"
    }

@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(await file.read())
    temp.close()

    cap = cv2.VideoCapture(temp.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    interval = max(fps, 1)

    probs = []
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if i % interval == 0:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            probs.append(predict_pil(img))
        i += 1

    cap.release()
    os.remove(temp.name)

    avg = sum(probs) / len(probs) if probs else 0.0

    return {
        "ai_probability": round(avg, 3),
        "frames_analyzed": len(probs),
        "label": "AI-generated" if avg > 0.6 else "Likely real"
    }

uvicorn main:app --host 0.0.0.0 --port 8000



implementation "androidx.compose.ui:ui:1.5.4"
implementation "androidx.activity:activity-compose:1.8.0"
implementation "com.squareup.retrofit2:retrofit:2.9.0"
implementation "com.squareup.retrofit2:converter-gson:2.9.0"
implementation "com.squareup.okhttp3:okhttp:4.11.0"
implementation "io.coil-kt:coil-compose:2.4.0"


interface ApiService {
    @Multipart
    @POST("detect/image")
    suspend fun detectImage(
        @Part file: MultipartBody.Part
    ): DetectionResponse

    @Multipart
    @POST("detect/video")
    suspend fun detectVideo(
        @Part file: MultipartBody.Part
    ): DetectionResponse
}

 object ApiClient {
    private const val BASE_URL = "http://YOUR_SERVER_IP:8000/"

    val api: ApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ApiService::class.java)
    }
}
 data class DetectionResponse(
    val ai_probability: Double,
    val frames_analyzed: Int? = null,
    val label: String
)

 @Composable
fun MainScreen(onPick: () -> Unit) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center
    ) {
        Text("AI Media Detector", fontSize = 26.sp)
        Spacer(Modifier.height(16.dp))
        Button(onClick = onPick) {
            Text("Select Image or Video")
        }
    }
}

 suspend fun uploadFile(
    context: Context,
    uri: Uri,
    isVideo: Boolean
): DetectionResponse {

    val input = context.contentResolver.openInputStream(uri)!!
    val file = File(context.cacheDir, if (isVideo) "video.mp4" else "image.jpg")

    file.outputStream().use { input.copyTo(it) }

    val body = file.asRequestBody(
        if (isVideo) "video/*".toMediaType() else "image/*".toMediaType()
    )

    val part = MultipartBody.Part.createFormData("file", file.name, body)

    return if (isVideo)
        ApiClient.api.detectVideo(part)
    else
        ApiClient.api.detectImage(part)
}

 @Composable
fun ResultScreen(response: DetectionResponse) {
    val percent = (response.ai_probability * 100).toInt()

    Column(Modifier.padding(24.dp)) {
        Text(response.label, fontSize = 22.sp)
        Spacer(Modifier.height(8.dp))
        Text("Confidence: $percent%")
        Spacer(Modifier.height(8.dp))
        Text(explain(response.ai_probability))
        Spacer(Modifier.height(16.dp))
        Text(
            "This result is probabilistic and may be incorrect.",
            fontSize = 12.sp
        )
    }
}

fun explain(p: Double): String = when {
    p > 0.85 -> "Strong indicators of AI generation."
    p > 0.65 -> "Moderate indicators of AI generation."
    p > 0.45 -> "Uncertain result."
    else -> "Likely real content."
}








# FinalProject

YOLOv5 기반 객체 변화 탐지 및 AI 틀린그림찾기 게임 프로젝트입니다.

사용자가 동일한 환경에서 촬영한 Before / After 이미지를 등록하고 실제 변경 사항을 입력하면, Edge System의 YOLOv5가 두 이미지의 객체를 검출하여 추가 및 제거된 객체를 분석합니다.  
Service System은 사용자가 입력한 실제 변경 사항과 AI 분석 결과를 비교하여 점수를 계산하고, Client System은 게임 진행 및 결과 확인을 위한 Android Native App UI를 제공합니다.

## Project Structure

```text
FinalProject/
├─ Edge_System/
│  ├─ detect.py
│  ├─ image_detector.py
│  ├─ difference_analyzer.py
│  ├─ game_worker.py
│  ├─ api_client.py
│  ├─ models/
│  ├─ utils/
│  ├─ data/
│  └─ requirements.txt
│
├─ Service_System/
│  ├─ manage.py
│  ├─ mysite/
│  ├─ blog/
│  ├─ game/
│  └─ requirements.txt
│
├─ Client_System/
│  ├─ app/
│  ├─ gradle/
│  ├─ gradlew
│  ├─ gradlew.bat
│  └─ settings.gradle.kts
│
├─ README.md
└─ url.txt
```

## System Architecture

### Edge System

- Python
- YOLOv5 pretrained model
- MS COCO 80-class object detection
- Before / After image object comparison
- Added / Removed object detection
- Django REST API communication

### Service System

- Python
- Django
- Django REST Framework
- Token Authentication
- Image Blog
- Game Session management
- Ground Truth management
- AI result management
- Score calculation
- PythonAnywhere deployment

### Client System

- Android Native App
- Java
- Before / After image upload
- Ground Truth input
- REST API communication
- Game result display
- Game history display

## Main Features

### F1. Pairwise Object Change Detection

YOLOv5 analyzes the Before and After images separately and compares the number of detected objects by class.

Example:

```text
Before
cell phone: 1
bottle: 0

After
cell phone: 0
bottle: 1

Result
REMOVED: cell phone
ADDED: bottle
```

### F2. Ground Truth and AI Result Comparison

The user directly enters the actual object changes.  
The Service System compares the user-provided Ground Truth with the AI analysis result.

### F3. AI vs User Scoring

The system calculates a game score based on how accurately the AI detected the actual changes.

A higher user score means the user successfully hid more changes from the AI.

## Edge System Setup

Move to the Edge System directory.

```bash
cd Edge_System
```

Install dependencies.

```bash
pip install -r requirements.txt
```

The project uses a YOLOv5 pretrained model.

Recommended weight:

```text
yolov5s.pt
```

The pretrained weight file is not included in this repository.

Run the Edge worker:

```bash
python game_worker.py
```

## Service System Setup

Move to the Service System directory.

```bash
cd Service_System
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Apply migrations.

```bash
python manage.py migrate
```

Run the local Django server.

```bash
python manage.py runserver
```

## Client System Setup

Open the following directory with Android Studio.

```text
Client_System/
```

Build and run the application on an Android Emulator or Android device.

## Environment Variables

The following environment variables may be required.

```text
DJANGO_SECRET_KEY
SERVER_URL
API_TOKEN
```

Do not commit actual secret values, passwords, tokens, or key files to this repository.

## Security

The following files and values must not be committed.

```text
.env
Django SECRET_KEY
API Token
Password
db.sqlite3
media files
YOLO weight files (*.pt)
Android local.properties
```

## Repository

Source Code:

```text
https://github.com/park1210/FinalProject
```
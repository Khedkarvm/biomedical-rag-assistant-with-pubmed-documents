# task 1
from fastapi import FastAPI
app = FastAPI()
@app.get("/")

def home():
    return {"message": "FastAPI is running"}


# task 2
import json
import aiofiles
import os

FILE_PATH = "intern_data.json"

async def read_intern_data():
    if not os.path.exists(FILE_PATH):
        return []
    
    async with aiofiles.open(FILE_PATH, "r") as file:
        content = await file.read()
        return json.loads(content) if content else []



async def write_intern_data(data):
    async with aiofiles.open(FILE_PATH, "w") as file:
        await file.write(json.dumps(data, indent=4))


# task 3

from fastapi import FastAPI
from pydantic import BaseModel
from utils.json_helper import read_intern_data, write_intern_data

app = FastAPI()

class InternCreate(BaseModel):
    name: str
    role: str

@app.post("/interns") 
async def add_intern(intern: InternCreate):
    interns = await read_intern_data()

    intern_id = len(interns) + 1

    new_intern = {
        "intern_id": intern_id,
        "name": intern.name,
        "role": intern.role,
        "status": "Active"
    }

    interns.append(new_intern)
    await write_intern_data(interns)

    return {
        "message": "Intern added successfully",
        "intern_id": intern_id
    }


# task 4

from typing import List
pp = FastAPI()

class ActivityInput(BaseModel):
    hours: int
    tasks: List[str]


@app.post("/interns/{intern_id}/activity")
async def add_activity(intern_id: str, activity: ActivityInput):
    interns = await read_intern_data()

    for intern in interns:
        if intern["intern_id"] == intern_id:
            if intern["status"] != "Active":
                return {"error": "Intern is not active"}

            intern["daily_hours"].append(activity.hours)
            intern["daily_tasks"].extend(activity.tasks)

            await write_intern_data(interns)
            return {"message": "Activity added successfully"}

    return {"error": "Intern not found"}


# task 5

@app.get("/interns/{intern_id}/summary")
async def view_intern_summary(intern_id: int):
    interns = await read_intern_data()

    for intern in interns:
        if intern["intern_id"] == intern_id:
            daily_hours = intern.get("daily_hours", [])
            total_hours = sum(daily_hours)
            average_hours = total_hours / len(daily_hours) if daily_hours else 0

            return {
                "intern_id": intern_id,
                "name": intern["name"],
                "role": intern["role"],
                "status": intern["status"],
                "total_hours": total_hours,
                "average_hours": average_hours,
                "daily_tasks": intern.get("daily_tasks", [])
            }

    return {"error": "Intern not found"}


# task 6 

@app.get("/statistics")
async def overall_statistics():
    interns = await read_intern_data()

    total_interns = len(interns)
    total_hours_all = 0
    top_performer = None
    max_hours = -1

    for intern in interns:
        daily_hours = intern.get("daily_hours", [])
        total_hours = sum(daily_hours)
        total_hours_all += total_hours

        if total_hours > max_hours:
            max_hours = total_hours
            top_performer = {
                "intern_id": intern["intern_id"],
                "name": intern["name"],
                "role": intern["role"],
                "total_hours": total_hours
            }

    average_hours = total_hours_all / total_interns if total_interns > 0 else 0

    return {
        "total_interns": total_interns,
        "average_hours": average_hours,
        "top_performer": top_performer
    }


# task 6

import asyncio

app = FastAPI()

class TestInput(BaseModel):
    delay_seconds: int

@app.post("/test-async")
async def test_async(input_data: TestInput):
    # Simulate an async task (like reading/writing JSON)
    await asyncio.sleep(input_data.delay_seconds)
    return {"message": f"Finished after {input_data.delay_seconds} seconds"}
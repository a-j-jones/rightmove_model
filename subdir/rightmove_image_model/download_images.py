from models import PropertyImages
from sqlmodel import create_engine, Session, select, func, distinct, or_
import asyncio
from httpx import AsyncClient
from typing import List
import re
from os import path
from tqdm import tqdm


async def download_image(url: str, filename: str, client: AsyncClient) -> None:
    """
    Download image from url and save it to filename
    :param url: url of the image
    :param filename: filename to save the image
    :return: None
    """
    response = await client.get(url)
    open(filename, "wb").write(response.content)


async def download_images() -> None:
    """
    Download a list of images concurrently
    :return: None
    """

    print("Connecting to database")
    sqlite_file_name = "database.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"
    engine = create_engine(sqlite_url, echo=False)
    pattern = re.compile(r'_(IMG_.*?)_')

    with Session(engine) as session:
        print("Creating httpx client")
        async with AsyncClient(timeout=5) as client:
            tasks = []
            print("Getting results from images table")
            statement = select(PropertyImages)
            results = session.exec(statement)
            count = 0
            print(f"Looping through properties")
            for image in tqdm(results):
                image_id = re.search(pattern, image.image_url).group(1)
                filename = f"images/{image.property_id}_{image_id}.jpg"
                if path.exists(filename):
                    continue
                tasks.append(download_image(image.image_url, filename, client))
                count += 1
                if count > 5000:
                    break

            print("Submitting URLs to asyc execution")
            await asyncio.gather(*tasks)

asyncio.run(download_images())

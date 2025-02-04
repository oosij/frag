import requests
import numpy as np
import pandas as pd
import json
import ast

import re
import os 
import time

from datetime import datetime, timedelta
from collections import defaultdict

import cx_Oracle

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, HasIdCondition, PointStruct


def get_all_points(client, collection_name, batch_size=100):
    all_points = []
    offset = 0  

    while True:
        result, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True
        )
        
        all_points.extend(result)

        if next_offset is None:
            break
        offset = next_offset

    return all_points

def date_to_timestamp(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").timestamp()



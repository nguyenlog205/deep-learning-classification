import pandas as pd
from pathlib import Path

def load_dataset(origin_path='./data/origin/'):
    base_path = Path(origin_path)
    data = []
    for label_dir in [x for x in base_path.iterdir() if x.is_dir()]:
        label = label_dir.name
        
        for video_file in [x for x in label_dir.iterdir() if x.is_file()]:
            data.append({
                'id': video_file.stem,
                'video_path': str(video_file),
                'label': label
            })

    return pd.DataFrame(data, columns=['id', 'video_path', 'label'])
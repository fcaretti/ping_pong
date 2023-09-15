from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import io
import os
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

from googleapiclient.discovery import build

def insert_image_to_sheet(spreadsheet_id, sheet_id, image_id, start_col, start_row,json_file='pypong.json'):
    creds = Credentials.from_service_account_file(json_file)
    sheet_service = build('sheets', 'v4', credentials=creds)

    requests = [{
        "insertImage": {
            "location": {
                "sheetId": sheet_id,
                "overlayPosition": {
                    "anchorCell": {
                        "rowIndex": start_row,
                        "columnIndex": start_col
                    },
                    "offsetXPixels": 0,
                    "offsetYPixels": 0
                },
            },
            "imageId": image_id,
        }
    }]
    body = {
        'requests': requests
    }
    response = sheet_service.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=body).execute()
    return response

def upload_image_and_delete_from_drive(filename,json_file='pypong.json'):
    creds = Credentials.from_service_account_file(json_file)
    drive_service = build('drive', 'v3', credentials=creds)

    # Upload the file
    file_metadata = {'name': filename}
    media = MediaFileUpload(filename, mimetype='image/png')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')

    # Delete the file after obtaining the file ID
    #drive_service.files().delete(fileId=file_id).execute()

    return file_id

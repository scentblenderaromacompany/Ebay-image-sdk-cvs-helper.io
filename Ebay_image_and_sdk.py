import os
import logging
import csv
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ExifTags
import pyheif
from datetime import datetime
import boto3
import json
from collections import Counter
import openai
from ebaysdk.trading import Connection as Trading
from ebaysdk.exception import ConnectionError

# Custom logging formatter with color
class CustomFormatter(logging.Formatter):
    green = "\033[92m"
    red = "\033[91m"
    reset = "\033[0m"

    FORMATS = {
        logging.INFO: green + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(levelname)s - %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logger.addHandler(handler)

SUPPORTED_IMAGE_TYPES = ['.heic', '.jpg', '.jpeg', '.png', '.tiff']
FONT_PATH = '/home/robertmcasper/Ebay_csv_project/Image_processing/fonts/GreatVibes-Regular.ttf'
DEFAULT_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TARGET_SIZE = (1024, 768)
THUMBNAIL_SIZE = (300, 300)
SKU_TRACKER_PATH = '/home/robertmcasper/Ebay_csv_project/sku_tracker.csv'

def initialize_metadata_file(metadata_path):
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w', newline='') as csvfile:
        fieldnames = ['SKU', 'ProductID', 'FileName', 'Type', 'Width', 'Height', 'Thumbnail', 'EXIF', 'Labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

def update_metadata_file(metadata_path, sku, product_id, file_name, image_type, width, height, thumbnail_name, exif_data, labels):
    with open(metadata_path, 'a', newline='') as csvfile:
        fieldnames = ['SKU', 'ProductID', 'FileName', 'Type', 'Width', 'Height', 'Thumbnail', 'EXIF', 'Labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'SKU': sku,
            'ProductID': product_id,
            'FileName': file_name,
            'Type': image_type,
            'Width': width,
            'Height': height,
            'Thumbnail': thumbnail_name,
            'EXIF': exif_data,
            'Labels': ', '.join(labels)
        })

def enhance_image(image):
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    enhanced_image = cv2.filter2D(image, -1, sharpen_kernel)
    return enhanced_image

def rotate_image(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def crop_image_to_center(image, target_width, target_height):
    width, height = image.size
    original_aspect = width / height
    target_aspect = target_width / target_height

    if original_aspect > target_aspect:
        new_width = int(height * target_aspect)
        left = (width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = height
    else:
        new_height = int(width / target_aspect)
        top = (height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = width

    return image.crop((left, top, right, bottom))

def add_watermark(image, text="Eternal Elegance Emporium", font_path=FONT_PATH):
    try:
        font = ImageFont.truetype(font_path, 32)  # Adjusted font size to 32 pixels
        logging.info(f"Using font from {font_path}")
    except IOError:
        logging.error("Font file not found or cannot be opened. Using default font.")
        try:
            font = ImageFont.truetype(DEFAULT_FONT_PATH, 32)  # Adjusted font size to 32 pixels
            logging.info("Using default font.")
        except IOError:
            logging.error("Default font file not found. Using default PIL font.")
            font = ImageFont.load_default()

    watermark = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(watermark, 'RGBA')
    width, height = image.size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x, y = width - text_width - 10, height - text_height - 10
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 200))  # Adjusted transparency to 200
    return Image.alpha_composite(image.convert('RGBA'), watermark)

def generate_thumbnail(image):
    thumbnail = image.copy()
    thumbnail.thumbnail(THUMBNAIL_SIZE)
    return thumbnail

def save_thumbnail(thumbnail, thumbnail_path):
    thumbnail.save(thumbnail_path, quality=95, optimize=True)
    logging.info(f"Saved thumbnail {thumbnail_path}")

def extract_exif_data(image):
    exif_data = {}
    try:
        info = image._getexif()
        if info is not None:
            for tag, value in info.items():
                key = ExifTags.TAGS.get(tag, tag)
                exif_data[key] = value
    except AttributeError:
        logging.error("Failed to extract EXIF data.")
    return exif_data

def convert_to_supported_format(file_path):
    try:
        image = Image.open(file_path)
        new_file_path = os.path.splitext(file_path)[0] + ".png"
        image.save(new_file_path)
        return new_file_path
    except UnidentifiedImageError as e:
        logging.error(f"Failed to convert {file_path}: {e}")
        return None

def extract_labels_from_image(image_path):
    client = boto3.client('rekognition')
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    response = client.detect_labels(
        Image={'Bytes': image_bytes},
        MaxLabels=10,
        MinConfidence=75
    )

    labels = [label['Name'] for label in response['Labels']]
    confidence = [label['Confidence'] for label in response['Labels']]
    label_confidence = dict(zip(labels, confidence))
    logging.info(f"Extracted labels: {label_confidence} for image {image_path}")
    return label_confidence

def write_labels_to_file(labels, product_dir):
    metadata_dir = os.path.join(product_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    labels_path = os.path.join(metadata_dir, "labels.txt")
    
    with open(labels_path, 'a') as f:
        for label, confidence in labels.items():
            f.write(f"{label}: {confidence:.2f}\n")
    
    logging.info(f"Written labels {list(labels.keys())} to {labels_path}")

def get_next_sku():
    if not os.path.exists(SKU_TRACKER_PATH):
        with open(SKU_TRACKER_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['SKU'])
    
    with open(SKU_TRACKER_PATH, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
    
    if len(data) <= 1:
        next_sku = 1
    else:
        next_sku = int(data[-1][0]) + 1
    
    with open(SKU_TRACKER_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([next_sku])
    
    return f'SKU{next_sku:05d}'

def process_image(file_path, output_dir, product_id, image_index, image_type, font_path, sku):
    try:
        if image_type == '.heic':
            heif_file = pyheif.read(file_path)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
        else:
            image = Image.open(file_path)

        image = rotate_image(image)
        exif_data = extract_exif_data(image)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        enhanced_image = enhance_image(image_cv)

        # Ensure the image is cropped only if necessary
        image = crop_image_to_center(image, TARGET_SIZE[0], TARGET_SIZE[1])

        resized_image = cv2.resize(enhanced_image, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        resized_image_pil = Image.fromarray(resized_image_rgb)
        resized_image_pil = add_watermark(resized_image_pil, font_path=font_path)

        product_dir = os.path.join(output_dir, f"Product_{product_id:05d}")
        os.makedirs(product_dir, exist_ok=True)
        output_file_name = f"{sku}_Image_{image_index:02d}.png"
        output_file_path = os.path.join(product_dir, output_file_name)
        resized_image_pil.save(output_file_path, quality=95, optimize=True)

        thumbnail = generate_thumbnail(resized_image_pil)
        thumbnail_dir = os.path.join(product_dir, "thumbnails")
        os.makedirs(thumbnail_dir, exist_ok=True)
        thumbnail_name = f"{sku}_Thumbnail_{image_index:02d}.png"
        thumbnail_path = os.path.join(thumbnail_dir, thumbnail_name)
        save_thumbnail(thumbnail, thumbnail_path)

        # Extract labels using Amazon Rekognition and write to file
        labels = extract_labels_from_image(file_path)
        write_labels_to_file(labels, product_dir)

        # Update the overall metadata CSV file
        metadata_path = os.path.join(output_dir, "metadata", "processed_images.csv")
        update_metadata_file(metadata_path, sku, product_id, output_file_name, image_type, resized_image_pil.width, resized_image_pil.height, thumbnail_name, exif_data, list(labels.keys()))

        logging.info(f"Processed {file_path} as {output_file_path}")
        return True, labels
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")
        return False, None

def create_summary_file(output_dir, product_count):
    summary_path = os.path.join(output_dir, "metadata", "summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_path, 'w') as summary_file:
        summary_file.write(f"Summary of Image Processing\n")
        summary_file.write(f"Date and Time: {current_time}\n")
        summary_file.write(f"Total Products Processed: {product_count}\n")
    logging.info(f"Summary file created at {summary_path}")

def process_directory(product_dir, output_dir, product_id, font_path):
    image_index = 1
    processed_count = 0
    all_labels = []

    # Generate SKU for the entire product folder
    sku = get_next_sku()

    for file in sorted(os.listdir(product_dir)):
        file_ext = os.path.splitext(file.lower())[1]
        if file_ext not in SUPPORTED_IMAGE_TYPES:
            file_path = convert_to_supported_format(os.path.join(product_dir, file))
            if file_path is None:
                continue
            file_ext = '.png'
        else:
            file_path = os.path.join(product_dir, file)

        success, labels = process_image(file_path, output_dir, product_id, image_index, file_ext, font_path, sku)
        if success:
            all_labels.extend(labels.keys())
            image_index += 1

    if image_index > 1:
        processed_count += 1

    # Save common labels to a summary file for this product
    product_metadata_dir = os.path.join(output_dir, f"Product_{product_id:05d}", "metadata")
    os.makedirs(product_metadata_dir, exist_ok=True)
    common_labels_path = os.path.join(product_metadata_dir, "common_labels.txt")
    common_labels = Counter(all_labels).most_common(10)
    with open(common_labels_path, 'w') as f:
        for label, count in common_labels:
            f.write(f"{label}: {count}\n")

    # Generate title and description using GPT-3
    title, description = generate_text_using_gpt3([label for label, _ in common_labels])

    # Generate eBay SDK listing template
    ebay_listing = create_ebay_listing_template(sku, title, description, common_labels)

    ebay_listing_path = os.path.join(product_metadata_dir, "ebay_listing.json")
    with open(ebay_listing_path, 'w') as f:
        json.dump(ebay_listing, f, indent=4)

    # Create an eBay listing
    create_ebay_listing(sku, title, description, common_labels)

    return processed_count

def generate_text_using_gpt3(labels):
    openai.api_key = "sk-proj-2UNxI3rifsdqLByVKT8ET3BlbkFJNt5fgDhxflv9mIt2X34I"
    prompt = f"Create an 80 character title and description for eBay listing based on these labels: {', '.join(labels)}"
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        max_tokens=150
    )
    response_text = response.choices[0].text.strip().split("\n")
    title = response_text[0].strip()
    description = response_text[1].strip() if len(response_text) > 1 else ""
    return title, description

def create_ebay_listing_template(sku, title, description, labels):
    item = {
        "SKU": sku,
        "Title": title,
        "Description": description,
        "Category": "Jewelry & Watches",  # Set your eBay category here
        "Condition": "New",
        "Price": "100.00",  # Set a default price, or make it dynamic
        "Quantity": "1",
        "Labels": [label for label, count in labels]
    }
    return item

def create_ebay_listing(sku, title, description, labels):
    try:
        api = Trading(config_file='ebay.yaml', warnings=True)
        item = {
            "Item": {
                "Title": title,
                "Description": description,
                "PrimaryCategory": {"CategoryID": "1234"},
                "StartPrice": "100.00",
                "ConditionID": 1000,
                "Country": "US",
                "Currency": "USD",
                "DispatchTimeMax": 3,
                "ListingDuration": "GTC",
                "ListingType": "FixedPriceItem",
                "PaymentMethods": "PayPal",
                "PayPalEmailAddress": "your-email@example.com",
                "PostalCode": "95125",
                "Quantity": 1,
                "ReturnPolicy": {
                    "ReturnsAcceptedOption": "ReturnsAccepted",
                    "RefundOption": "MoneyBack",
                    "ReturnsWithinOption": "Days_30",
                    "Description": "If you are not satisfied, return the item for refund.",
                    "ShippingCostPaidByOption": "Buyer"
                },
                "ShippingDetails": {
                    "ShippingServiceOptions": [{
                        "ShippingServicePriority": 1,
                        "ShippingService": "USPSMedia"
                    }]
                },
                "Site": "US"
            }
        }
        response = api.execute('AddItem', item)
        logging.info(response.dict())
    except ConnectionError as e:
        logging.error(e)
        logging.error(e.response.dict())

def upload_to_s3(bucket_name, local_path, s3_path):
    s3_client = boto3.client('s3')

    for root, dirs, files in os.walk(local_path):
        for file in files:
            full_local_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_local_path, local_path)
            s3_file_path = os.path.join(s3_path, relative_path)
            
            s3_client.upload_file(full_local_path, bucket_name, s3_file_path)
            logging.info(f"Uploaded {full_local_path} to s3://{bucket_name}/{s3_file_path}")

def create_ebay_csv(output_dir):
    ebay_csv_path = os.path.join(output_dir, "metadata", "ebay_upload.csv")
    with open(ebay_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['SKU', 'Title', 'Item Description', 'Category', 'Condition', 'Price', 'Quantity', 'Image URL', 'Thumbnail URL', 'Labels']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        processed_images_path = os.path.join(output_dir, "metadata", "processed_images.csv")
        with open(processed_images_path, 'r') as processed_file:
            reader = csv.DictReader(processed_file)
            for row in reader:
                title = f"{row['Labels']}"  # Customize title based on labels
                description = f"High-quality {row['Labels']}"  # Customize description based on labels
                category = "Jewelry & Watches"  # Set your eBay category here
                condition = "New"
                price = "100.00"  # Set a default price, or make it dynamic
                quantity = "1"
                image_url = f"https://s3://eternal-elegance-emporium-image-bucket/ebayimages/processed/completed//{row['SKU']}_Image_01.png"
                thumbnail_url = f"https://s3://eternal-elegance-emporium-image-bucket/ebayimages/processed/completed//{row['SKU']}_Thumbnail_01.png"

                writer.writerow({
                    'SKU': row['SKU'],
                    'Title': title,
                    'Item Description': description,
                    'Category': category,
                    'Condition': condition,
                    'Price': price,
                    'Quantity': quantity,
                    'Image URL': image_url,
                    'Thumbnail URL': thumbnail_url,
                    'Labels': row['Labels']
                })

    logging.info(f"eBay CSV file created at {ebay_csv_path}")

def create_ebay_json(output_dir):
    ebay_json_path = os.path.join(output_dir, "metadata", "ebay_upload.json")
    ebay_json_data = []

    processed_images_path = os.path.join(output_dir, "metadata", "processed_images.csv")
    with open(processed_images_path, 'r') as processed_file:
        reader = csv.DictReader(processed_file)
        for row in reader:
            item = {
                "SKU": row['SKU'],
                "Title": f"{row['Labels']}",  # Customize title based on labels
                "Description": f"High-quality {row['Labels']}",  # Customize description based on labels
                "Category": "Jewelry & Watches",  # Set your eBay category here
                "Condition": "New",
                "Price": "100.00",  # Set a default price, or make it dynamic
                "Quantity": "1",
                "ImageURL": f"https://s3://eternal-elegance-emporium-image-bucket/ebayimages/processed/completed//{row['SKU']}_Image_01.png",
                "ThumbnailURL": f"https://s3://eternal-elegance-emporium-image-bucket/ebayimages/processed/completed//{row['SKU']}_Thumbnail_01.png",
                "Labels": row['Labels']
            }
            ebay_json_data.append(item)
    
    with open(ebay_json_path, 'w') as json_file:
        json.dump(ebay_json_data, json_file, indent=4)

    logging.info(f"eBay JSON file created at {ebay_json_path}")

def main():
    input_dir = '/home/robertmcasper/Ebay_csv_project/Image_processing/input/raw/June 12 2024'
    base_output_dir = '/home/robertmcasper/Ebay_csv_project/Image_processing/output/processed'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"Processed_{timestamp}")
    metadata_path = os.path.join(output_dir, "metadata", "processed_images.csv")
    font_path = '/home/robertmcasper/Ebay_csv_project/Image_processing/fonts/GreatVibes-Regular.ttf'

    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    initialize_metadata_file(metadata_path)
    product_id = 1
    processed_product_count = 0

    dirs = [os.path.join(root, d) for root, dnames, _ in os.walk(input_dir) for d in sorted(dnames)]
    
    for d in dirs:
        try:
            processed_count = process_directory(d, output_dir, product_id, font_path)
            processed_product_count += processed_count
            product_id += 1
        except Exception as e:
            logging.error(f"Failed to process directory {d}: {e}")

    create_summary_file(output_dir, processed_product_count)
    create_ebay_csv(output_dir)
    create_ebay_json(output_dir)
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
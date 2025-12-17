from ultralytics import YOLO

def main():
    # === 1. Load model YOLOv8n Pretrained ===
    model = YOLO("yolov8n.pt")  # versi nano

    # === 2. Train model ===
    model.train(
        data='C:/Users/trkb/Documents/Skripsi_Panji/data.yaml',   
        epochs=200,                   
        batch=-1,                      
        imgsz=640,                    
        device=0,                     
        workers=8,                    
        cache=False,                  
        save=True,                    
        verbose=True,                 
        project='C:/Users/trkb/Documents/Skripsi_Panji/yolo_mosaic_off',   
        name='yolov8n_mosaic_off',           
        mosaic=0.0,
        patience=30                 
    )

    # === 3. Evaluate model setelah training ===
    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    main()

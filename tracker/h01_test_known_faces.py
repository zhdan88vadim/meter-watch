import face_recognition
import cv2
import os

def test_face_images():
    """Test all images in known_faces directory"""
    # known_dir = "known_faces"
    known_dir = "try_to_find_face"
    
    if not os.path.exists(known_dir):
        print(f"❌ Directory '{known_dir}' not found!")
        return
    
    print(f"📂 Testing images in {known_dir}:")
    
    for filename in os.listdir(known_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(known_dir, filename)
            print(f"\n📷 Testing: {filename}")
            
            # Load and test
            image = face_recognition.load_image_file(image_path)

            # scaled_image = cv2.resize(image, (0,0), fx=3.0, fy=3.0)
            # face_locations = face_recognition.face_locations(scaled_image)

            # face_locations = face_recognition.face_locations(image)
            # face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=2) 
            face_locations = face_recognition.face_locations(image, model="cnn")
            
            if face_locations:
                print(f"   ✅ Found {len(face_locations)} face(s)")
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    print(f"      Face {i+1}: top={top}, right={right}, bottom={bottom}, left={left}")

                    # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                    # cv2.imwrite(f"marked_{filename}", image)

            else:
                print(f"   ❌ No faces detected!")
                print(f"      Image dimensions: {image.shape}")
                print(f"      Try: photos with clear front-facing faces, good lighting")

if __name__ == "__main__":
    test_face_images()
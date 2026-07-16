import face_recognition
import cv2
import os
import pickle

def load_known_faces(known_dir="known_faces"):
    """Load all known face encodings from a directory"""
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(known_dir):
        print(f"❌ Directory '{known_dir}' not found!")
        return known_face_encodings, known_face_names
    
    print(f"📂 Loading known faces from {known_dir}:")
    
    for filename in os.listdir(known_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(known_dir, filename)
            print(f"   Loading: {filename}")
            
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Get face locations (use CNN for better detection)
            # face_locations = face_recognition.face_locations(image, model="cnn")
            face_locations = face_recognition.face_locations(image, model="hog")
            
            if face_locations:
                # Get face encodings
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                if face_encodings:
                    # Use first face found
                    known_face_encodings.append(face_encodings[0])
                    # Use filename without extension as name
                    name = os.path.splitext(filename)[0]
                    known_face_names.append(name)
                    print(f"      ✅ Loaded: {name}")
                else:
                    print(f"      ❌ Could not encode face")
            else:
                print(f"      ❌ No face detected in {filename}")
    
    print(f"\n✅ Loaded {len(known_face_encodings)} known faces")
    return known_face_encodings, known_face_names

def recognize_faces_in_image(image_path, known_face_encodings, known_face_names):
    """Recognize faces in a single image"""
    # Load the test image
    image = face_recognition.load_image_file(image_path)
    
    # Find all faces in the image
    # face_locations = face_recognition.face_locations(image, model="cnn")
    face_locations = face_recognition.face_locations(image, model="hog")
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    results = []
    
    for face_encoding in face_encodings:
        # Compare with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"
        
        # If a match was found
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            
            # Optional: Get confidence score
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            confidence = 1 - face_distances[best_match_index]
            
            results.append({
                'name': name,
                'confidence': confidence,
                'matched': True
            })
        else:
            results.append({
                'name': 'Unknown',
                'confidence': 0,
                'matched': False
            })
    
    return face_locations, results

def test_face_recognition():
    """Main function to test face recognition"""
    # Load known faces
    known_encodings, known_names = load_known_faces("known_faces")
    
    if not known_encodings:
        print("❌ No known faces loaded! Please add images to 'known_faces' directory.")
        return
    
    # Directory to test
    test_dir = "try_to_find_face"
    
    if not os.path.exists(test_dir):
        print(f"❌ Directory '{test_dir}' not found!")
        return
    
    print(f"\n🔍 Testing images in {test_dir}:\n")
    
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_dir, filename)
            print(f"📷 Analyzing: {filename}")
            
            # Recognize faces
            face_locations, results = recognize_faces_in_image(
                image_path, 
                known_encodings, 
                known_names
            )
            
            if face_locations:
                print(f"   ✅ Found {len(face_locations)} face(s):")
                for i, (location, result) in enumerate(zip(face_locations, results)):
                    top, right, bottom, left = location
                    if result['matched']:
                        confidence_pct = result['confidence'] * 100
                        print(f"      Face {i+1}: {result['name']} (confidence: {confidence_pct:.1f}%)")
                        print(f"         Position: top={top}, right={right}, bottom={bottom}, left={left}")
                    else:
                        print(f"      Face {i+1}: ❌ Unknown person")
                        print(f"         Position: top={top}, right={right}, bottom={bottom}, left={left}")
            else:
                print(f"   ❌ No faces detected in image!")
                # print(f"      Image dimensions: {image.shape}")

if __name__ == "__main__":
    test_face_recognition()
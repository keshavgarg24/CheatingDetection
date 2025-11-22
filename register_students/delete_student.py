"""
Delete Student from Face Database and Monitoring Database

This script allows administrators to delete a student from both:
- student_faces.sqlite (face recognition database)
- monitoring.sqlite (monitoring and violations database)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from register_students.student_registration import StudentFaceDatabase
from db import delete_student as delete_from_monitoring_db

def main():
    """Main function to delete a student"""
    print("=" * 60)
    print("DELETE STUDENT")
    print("=" * 60)
    print()
    
    # Initialize face database
    face_db = StudentFaceDatabase()
    
    # List all registered students first
    students = face_db.list_registered_students()
    
    if not students:
        print("[INFO] No registered students found in database.")
        return
    
    print(f"Found {len(students)} registered student(s):")
    print("-" * 60)
    for i, student in enumerate(students, 1):
        print(f"{i}. ID: {student['student_id']}, Name: {student['name']}, "
              f"Roll: {student['roll_number']}, Seat: {student['seat_number']}")
    print("-" * 60)
    print()
    
    # Get student ID to delete
    student_id = input("Enter the Student ID to delete: ").strip()
    
    if not student_id:
        print("[ERROR] Student ID cannot be empty!")
        return
    
    # Verify student exists
    student_info = face_db.get_student(student_id)
    if not student_info:
        print(f"[ERROR] Student with ID '{student_id}' not found in face database!")
        return
    
    # Confirm deletion
    print()
    print(f"Student to delete:")
    print(f"  ID: {student_info['student_id']}")
    print(f"  Name: {student_info['name']}")
    print(f"  Roll Number: {student_info['roll_number']}")
    print(f"  Seat Number: {student_info['seat_number']}")
    print()
    
    confirm = input("Are you sure you want to delete this student? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("[CANCELLED] Deletion cancelled.")
        return
    
    print()
    print("[INFO] Deleting student from databases...")
    print("-" * 60)
    
    # Delete from face database
    face_deleted = face_db.delete_student(student_id)
    
    # Delete from monitoring database
    monitoring_deleted = delete_from_monitoring_db(student_id)
    
    print("-" * 60)
    
    if face_deleted:
        print(f"[SUCCESS] Student {student_id} deleted from face database")
    else:
        print(f"[WARNING] Could not delete {student_id} from face database (may not exist)")
    
    if monitoring_deleted:
        print(f"[SUCCESS] Student {student_id} deleted from monitoring database")
    else:
        print(f"[INFO] Student {student_id} not found in monitoring database (may not have been monitored yet)")
    
    print()
    print("=" * 60)
    print("DELETION SUMMARY")
    print("=" * 60)
    
    # Show remaining students
    remaining_students = face_db.list_registered_students()
    print(f"\nRemaining registered students: {len(remaining_students)}")
    if remaining_students:
        print("-" * 60)
        for student in remaining_students:
            print(f"ID: {student['student_id']}, Name: {student['name']}, "
                  f"Roll: {student['roll_number']}, Seat: {student['seat_number']}")
    else:
        print("[INFO] No registered students remaining.")
    
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Operation cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback
        traceback.print_exc()

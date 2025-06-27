from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class StudentDataRetriever:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.grade_order = {'A+': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4}  # Add this
        
    def get_relevant_data(self, question, student_data):
        # Handle grade-specific questions first
        if 'best grade' in question.lower() or 'highest grade' in question.lower():
            student_data['grade_rank'] = student_data['grade'].map(self.grade_order)
            top_student = student_data.sort_values('grade_rank').head(1)
            return top_student.to_string(index=False)
            
        # Original TF-IDF logic for other questions
        student_data['text_representation'] = student_data.apply(
            lambda row: f"Name: {row['name']}, Age: {row['age']}, " +
                       f"Math: {row['math']}, Science: {row['science']}, " +
                       f"English: {row['english']}, History: {row['history']}, " +
                       f"Grade: {row['grade']}",
            axis=1
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(student_data['text_representation'])
        question_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-3:][::-1]
        return student_data.iloc[top_indices].to_string(index=False)
    
class TutorDataRetriever:
        def __init__(self):
            self.vectorizer = TfidfVectorizer()

        def get_relevant_data(self, question, tutor_data):
            tutor_data['text_representation'] = tutor_data.apply(
                lambda row: f"Name: {row['name']}, Subject: {row['subject']}, " +
                            f"Experience: {row['experience']} years, Rating: {row['rating']}",
                axis=1
            )
            tfidf_matrix = self.vectorizer.fit_transform(tutor_data['text_representation'])
            question_vec = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
            top_indices = similarities.argsort()[-3:][::-1]
            return tutor_data.iloc[top_indices].to_string(index=False)

class DataRetriever:
    def __init__(self):
        self.student_retriever = StudentDataRetriever()
        self.tutor_retriever = TutorDataRetriever()

    def get_relevant_data(self, question, data, data_type):
        if data_type == 'student':
            return self.student_retriever.get_relevant_data(question, data)
        elif data_type == 'tutor':
            return self.tutor_retriever.get_relevant_data(question, data)
        else:
            raise ValueError("Invalid data_type. Use 'student' or 'tutor'.")
        


@dataclass
class Requirements:
private:
    title: str
    job_desc: str
    industry: str
    location: str 
    yrs_experience: int
    must_have_skills: Collection[str]
    nice_to_have_skills: Collection[str]

public: 
    def __init__(self, requirements):
        self.title = requirements['title']
        self.job_desc = requirements['job_desc']
        self.industry = requirements['industry']
        self.location = requirements['location']
        self.yrs_experience = requirements['yrs_experience']
        self.must_have_skills = requirements['must_have_skills']
        self.nice_to_have_skills = requirements['nice_to_have_skills']
    
    def to_json(self):
        return {
            'title': self.title,
            'job_desc': self.job_desc,
            'industry': self.industry,
            'location': self.location,
            'yrs_experience': self.yrs_experience,
            'must_have_skills': self.must_have_skills,
            'nice_to_have_skills': self.nice_to_have_skills
        }

@dataclass
class Candidate:
private:
    name: str
    email: str
    yrs_experience: int
    skills: Collection[str]
    industry: str
    location: str
    handles: Dict[str, str]
    
public: 
    def __init__(self, candidate):
        self.name = candidate['name']
        self.email = candidate['email']
        self.yrs_experience = candidate['yrs_experience']
        self.skills = candidate['skills']
        self.industry = candidate['industry']
        self.location = candidate['location']
        self.handles = candidate['handles']
    
    def to_json(self):
        return {
            'name': self.name,
            'email': self.email,
            'yrs_experience': self.yrs_experience,
            'skills': self.skills,
            'industry': self.industry,
            'location': self.location,
            'handles': self.handles
        }

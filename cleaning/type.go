package cleaning

import "go.mongodb.org/mongo-driver/bson/primitive"

type Dataset struct {
	ID        primitive.ObjectID `json:"id" bson:"_id,omitempty"`
	Questions string             `json:"questions" bson:"questions"`
	Answers   string             `json:"answer" bson:"answer"`
}

type PaginationParams struct {
	Page   int64 `json:"page"`
	Limit  int64 `json:"limit"`
	Offset int64 `json:"offset"`
}

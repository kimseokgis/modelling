package cleaning

import (
	"context"
	"errors"
	"fmt"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"log"
	"os"
)

func DBAdmn(dbname string) *mongo.Database {
	connectionstr := os.Getenv("DBADMN")
	if connectionstr == "" {
		panic(fmt.Errorf("DBADMN ENV NOT FOUND"))
	}
	clay, err := mongo.Connect(context.TODO(), options.Client().ApplyURI(connectionstr))
	if err != nil {
		panic(fmt.Errorf("MongoConnect: %+v \n", err))
	}
	return clay.Database(dbname)
}

func GetAllDatasets(ctx context.Context, params PaginationParams, db *mongo.Database) ([]Dataset, error) {
	if params.Page < 1 {
		return nil, fmt.Errorf("Page number must be greater than 0")
	}
	if params.Limit < 1 {
		return nil, fmt.Errorf("Page size must be greater than 0")
	}

	collection := db.Collection("datasets")
	filter := bson.M{}

	findOptions := options.Find()
	findOptions.SetLimit(params.Limit)
	findOptions.SetSkip(params.Page*params.Limit - params.Limit)

	cur, err := collection.Find(ctx, filter, findOptions)
	if err != nil {
		return nil, fmt.Errorf("Find: %+v \n", err)
	}
	defer cur.Close(context.TODO())

	var datasets []Dataset
	for cur.Next(context.TODO()) {
		var dataset Dataset
		err := cur.Decode(&dataset)
		if err != nil {
			log.Printf("Decode: %+v \n", err)
			continue
		}
		datasets = append(datasets, dataset)
	}

	if err := cur.Err(); err != nil {
		return nil, fmt.Errorf("Cursor: %+v \n", err)
	}

	defer cur.Close(context.TODO())
	return datasets, nil
}

func GetDatasetByID(id primitive.ObjectID, db *mongo.Database) (*Dataset, error) {
	collection := db.Collection("datasets")
	filter := bson.D{{"_id", id}}
	var dataset Dataset
	err := collection.FindOne(context.TODO(), filter).Decode(&dataset)
	if err != nil {
		if errors.Is(err, mongo.ErrNoDocuments) {
			return nil, fmt.Errorf("Dataset with _id '%s' not found", id.Hex())
		}
		return nil, fmt.Errorf("FindOne: %+v", err)
	}
	return &dataset, nil
}

func UpdateDatasetByID(id primitive.ObjectID, updateData Dataset, db *mongo.Database) (err error) {
	collection := db.Collection("datasets")
	filter := bson.M{"_id": id}
	update := bson.M{
		"$set": updateData,
	}
	err = collection.FindOneAndUpdate(context.TODO(), filter, update).Err()
	if err != nil {
		return fmt.Errorf("UpdateOne: %+v", err)
	}
	return nil
}

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

// Fungsi untuk mengambil semua dokumen dari koleksi "dataset"
func GetAllDatasets(ctx context.Context) ([]Dataset, error) {
	db := DBAdmn("AI")
	collection := db.Collection("datasets")

	// Membuat filter kosong untuk mengambil semua dokumen
	filter := bson.D{}

	// Melakukan query untuk mengambil semua dokumen
	cur, err := collection.Find(ctx, filter)
	if err != nil {
		return nil, fmt.Errorf("Find: %+v \n", err)
	}
	defer cur.Close(context.TODO())

	// Mengambil hasil query dan memasukkannya ke slice Dataset
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

	return datasets, nil
}

func GetDatasetByID(id primitive.ObjectID) (*Dataset, error) {
	db := DBAdmn("AI")
	collection := db.Collection("datasets")

	// Membuat filter berdasarkan _id
	filter := bson.D{{"_id", id}}

	// Melakukan query untuk mengambil dokumen dengan _id tertentu
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

func UpdateDatasetByID(id primitive.ObjectID, updateData Dataset) error {
	db := DBAdmn("AI")
	collection := db.Collection("datasets")

	// Membuat filter berdasarkan _id
	filter := bson.D{{"_id", id}}

	// Membuat opsi untuk mengatur opsi dokumen yang diperbarui
	opts := options.Update().SetUpsert(false)

	// Membuat operasi update
	update := bson.D{
		{"$set", updateData},
	}

	// Melakukan update dokumen
	_, err := collection.UpdateOne(context.TODO(), filter, update, opts)
	if err != nil {
		return fmt.Errorf("UpdateOne: %+v", err)
	}

	return nil
}

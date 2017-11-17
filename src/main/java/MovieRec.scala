/**
  * Created by itim on 11.11.2017.
  */

import java.io.{File, PrintWriter}

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.log4j.{Level, Logger}

import scala.io._

object MovieRec {

  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)

  case class Movie(movieId: Int, movieName: String, rating: Float)

  case class UserMovie(userId: Int, movieId: Int)


  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    return Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    val sparkSession = SparkSession
      .builder()
      .appName("spark-read-csv")
      .master("spark://10.8.41.146:7077")
      .getOrCreate();

    import sparkSession.implicits._

    // load ratings and movie titles
    val movieLensHomeDir = args(0)
    sparkSession.sparkContext.addJar(movieLensHomeDir + "MovieRec")
    //Load Ratings
    val ratings = sparkSession
      .read.textFile(movieLensHomeDir + "Ratings")
      .map(parseRating)
      .toDF()

    val predictRDD = sparkSession
      .read.textFile(movieLensHomeDir + "Predictions").map { line =>
      val fields = line.split("::")
      UserMovie(fields(0).toInt, fields(1).toInt)
    }

    //Load Movies
    //    val moviesRDD = sparkSession
    //      .read.textFile(movieLensHomeDir + "movies.dat").map { line =>
    //      val fields = line.split("::")
    //      (fields(0).toInt, fields(1))
    //    }
    //    ratings.show(10)
    //    moviesRDD.show(10)
    val myRating = sparkSession.read.textFile(movieLensHomeDir + "PersonalRatings")
      .map(parseRating)
      .toDF()

    val ratingWithMyRats = ratings.union(myRating)
    val Array(training, test) = ratingWithMyRats.randomSplit(Array(0.5, 0.5))

    val als = new ALS()
      .setMaxIter(3)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")

    //Get trained model
    val model = als.fit(training)

    val predictions = model.transform(test).na.drop

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)

    //println(s"Root-mean-square error = $rmse")

    val predict = predictRDD.toDF()
    val myPredictions = model.transform(predict).na.drop
    //Get My Predictions
    //val myPredictions = model.transform(myRating).na.drop
    //myPredictions.show(15)
    //val writer = new PrintWriter(new File(movieLensHomeDir+"Res"))
    myPredictions.repartition(1).write.format("csv").save(movieLensHomeDir + "mymovies")
    //myPredictions.map(x=>x.mkString).rdd.saveAsTextFile(movieLensHomeDir+"Res")
  }
}

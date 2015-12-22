package org.apache.spark.examples.mllib;

import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public final class JavaKMeans {

  private static class ParsePoint implements Function<String, Vector> {
    private static final Pattern SPACE = Pattern.compile(" ");
    // set up patterns
    @Override
    public Vector call(String line) {
      String[] tok = SPACE.split(line); // split the file
      double[] point = new double[tok.length]; 
      for (int i = 0; i < tok.length; ++i) { 
        point[i] = Double.parseDouble(tok[i]);
      }
      return Vectors.dense(point);  // save features of points into array
    }
  }

  public static void main(String[] args) {
    if (args.length < 3) {
      System.err.println(
        "Usage: JavaKMeans <input_file> <k> <max_iterations> [<runs>]");
      System.exit(1);
    }
    // if not enough argument, exit
    String inputFile = args[0];
    int k = Integer.parseInt(args[1]);
    int iterations = Integer.parseInt(args[2]);
    int runs = 1;
    // take in the arguments

    if (args.length >= 4) {
      runs = Integer.parseInt(args[3]);
    }

    SparkConf sparkConf = new SparkConf().setAppName("JavaKMeans");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);
    
    // set up the spark context

    JavaRDD<String> lines = sc.textFile(inputFile);
    // split input file into lines and transform to RDD
    JavaRDD<Vector> points = lines.map(new ParsePoint());
    // split lines into points and map them into a new RDD
    KMeansModel model = KMeans.train(points.rdd(), k, iterations, runs, KMeans.K_MEANS_PARALLEL());
    // build the k-means model with iterations

    System.out.println("Cluster centers:");
    for (Vector center : model.clusterCenters()) {
      System.out.println(" " + center);
    }
    double cost = model.computeCost(points.rdd());
    System.out.println("Cost: " + cost);
    //print out the results.
    sc.stop();
  }
}

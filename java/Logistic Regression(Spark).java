package org.apache.spark.examples.mllib;

import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;


public final class JavaLR {

  static class ParsePoint implements Function<String, LabeledPoint> {
    private static final Pattern COMMA = Pattern.compile(",");
    private static final Pattern SPACE = Pattern.compile(" ");

    @Override
    public LabeledPoint call(String line) {
      String[] parts = COMMA.split(line);
      double y = Double.parseDouble(parts[0]); // get the y variable
      String[] tok = SPACE.split(parts[1]);  // get the features
      double[] x = new double[tok.length];
      for (int i = 0; i < tok.length; ++i) {
        x[i] = Double.parseDouble(tok[i]);
      }
      return new LabeledPoint(y, Vectors.dense(x));  // parse point into y,x,x,x,x,x
    }
  }
  // parse the points into array

  public static void main(String[] args) {
    if (args.length != 3) {
      System.err.println("Usage: LR <input_dir> <step_size> <niters>");
      System.exit(1);
      // if not enough argument, exit
    }
    SparkConf sparkConf = new SparkConf().setAppName("JavaLR");
    JavaSparkContext sc = new JavaSparkContext(sparkConf);
    // initialize set up spark context

    JavaRDD<String> lines = sc.textFile(args[0]);
    // split the file  into line and transform RDD
    JavaRDD<LabeledPoint> points = lines.map(new ParsePoint()).cache();
    // split the line into points and save the transformed points RDD in memory
    double stepSize = Double.parseDouble(args[1]);
    int iterations = Integer.parseInt(args[2]);

    //take in the argument

    LogisticRegressionModel model = LogisticRegressionWithSGD.train(points.rdd(),
      iterations, stepSize);
    // build the model and predict

    System.out.print("Final w: " + model.weights());
    // output the model weight. 
    sc.stop();
  }
}

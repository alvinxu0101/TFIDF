package com.alvinxu.tfidf

import org.apache.spark.{SparkConf, SparkContext}

object TFIDFScala {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("特征值提取").setMaster("local[2]")
    val sc = new SparkContext(conf)
  }
}

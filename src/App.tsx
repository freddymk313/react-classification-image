import React, { useEffect, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import { BsUpload } from "react-icons/bs";

const App: React.FC = () => {
  const [model, setModel] = useState<tf.GraphModel | null>(null);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loadingModel, setLoadingModel] = useState<boolean>(false);
  const [loadingPrediction, setLoadingPrediction] = useState<boolean>(false);
  const [message, setMessage] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [testImg, setTestImg] = useState<string | null>(null);
  const [predictionPercentage, setPredictionPercentage] = useState<
    number | null
  >(null);

  useEffect(() => {
    // loadModel();
  }, []);

  const loadModel = async () => {
    try {
      setLoadingModel(true);
      const loadedModel = await tf.loadGraphModel("../public/model/model.json");
      setModel(loadedModel);
      setLoadingModel(false);
      setMessage("Model loaded");
    } catch (error) {
      console.error("Erreur lors du chargement du modèle:", error);
      setError("Erreur lors du chargement du modèle");
      setLoadingModel(false);
    }
  };

  const handleStart = async () => {
    setLoadingPrediction(true);
    if (!model) {
      console.error("Le modèle n'est pas encore chargé.");
      return;
    }

    if (!testImg) {
      console.error("Aucune image sélectionnée.");
      return;
    }

    // Charger l'image
    const img = new Image();
    img.src = testImg;

    img.onload = async () => {
      try {
        // Prétraiter l'image
        const tensorImg = tf.browser
          .fromPixels(img)
          .resizeNearestNeighbor([224, 224])
          .toFloat();
        const normalizedImg = tensorImg.div(tf.scalar(255));

        // Ajouter une dimension pour correspondre aux dimensions attendues par le modèle
        const expandedImg = normalizedImg.expandDims(0);

        // Faire la prédiction
        const predictions = (await model.predict(expandedImg)) as tf.Tensor;
        const predictionsArray = (await predictions.data()) as Float32Array;

        // Récupérer la classe prédite
        const classNames = [
          "Ali",
          "Arnold",
          "Freddy",
          "Gabriel",
          "Gael",
          "Gerard",
          "Jean-Luc",
          "Jules",
          "Nathan",
          "Paul",
        ];
        const predictedClassIndex = predictionsArray.indexOf(
          Math.max(...predictionsArray)
        );
        const predictedClass = classNames[predictedClassIndex];
        setPrediction(predictedClass);

        const predictedProbability =
          predictionsArray[predictedClassIndex] * 100;
        setPredictionPercentage(predictedProbability);

        setLoadingPrediction(false);
      } catch (error) {
        console.error("Erreur lors de la prédiction:", error);
        setError("Erreur lors de la prédiction");
        setLoadingPrediction(false);
      }
    };
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files && event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (reader.result) {
          const imageData = reader.result as string;
          setTestImg(imageData);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <main className="min-h-screen flex items-center justify-center bg-gray-950">
      <div className="">
        <div className="h-60 w-60 border border-gray-950 rounded-lg">
          <div className="mb-3">
            {message && (
              <p className="text-green-500 text-center text-sm">{message}</p>
            )}
            {error && (
              <p className="text-red-500 text-center text-sm">{error}</p>
            )}
          </div>

          {!testImg ? (
            <div className="h-60 w-60 flex flex-col items-center justify-center border rounded-lg">
              <label htmlFor="file-upload" className="cursor-pointer">
                <BsUpload size={32} className="text-gray-50" />
                <input
                  id="file-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </label>
              <span className="text-sm text-gray-50 mt-0.5">Upload image</span>
            </div>
          ) : (
            <img
              src={testImg}
              alt="image_test"
              className="rounded-lg h-60 w-60 object-cover"
            />
          )}
        </div>

        <div className="mt-[60px]">
          <button
            onClick={handleStart}
            className="py-3 px-6 bg-green-50 text-gray-950 w-full rounded-lg hover:bg-gray-50/85 transition"
            disabled={loadingModel || !testImg}
          >
            {loadingPrediction ? "Loading..." : "Prediction"}
          </button>
        </div>

        {prediction && (
          <div className="mt-2">
            <p className="text-center text-gray-50">
              Predicted class:{" "}
              <span className="text-green-500 underline">{prediction}</span>
            </p>

            <p className="text-center text-gray-50 mt-1">
              Prediction probability:{" "}
              <span className="text-green-500">
                {parseFloat(predictionPercentage).toFixed(2)}%
              </span>
            </p>
          </div>
        )}
      </div>
    </main>
  );
};

export default App;

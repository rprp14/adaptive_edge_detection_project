// Fetch latest uploaded image and processed outputs
window.onload = function () {
  fetch("http://127.0.0.1:5000/get_latest_image")
    .then((response) => response.json())
    .then((data) => {
      console.log("Fetched data:", data);

      document.getElementById("uploadedImage").src = data.file_path;
      document.getElementById("basicOutput").src = data.basic_output;
      document.getElementById("adaptiveOutput").src = data.adaptive_output;
      document.getElementById("sobelOutput").src = data.sobel_output;
      document.getElementById("psoOutput").src = data.pso_output;
      document.getElementById("mlOutput").src = data.ml_output;
      document.getElementById("vitOutput").src = data.vit_output;
      document.getElementById("vitPsoOutput").src = data.vit_pso_output;
    })
    .catch((error) => console.error("Error fetching image:", error));
};

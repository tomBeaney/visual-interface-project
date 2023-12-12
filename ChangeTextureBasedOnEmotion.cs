using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;

public class ChangeTextureBasedOnEmotion : MonoBehaviour
{
    public Renderer modelRenderer; // Reference to the renderer of the model
    public Texture happyTexture; // Texture for the happy emotion
    public Texture sadTexture; // Texture for the sad emotion
    public Texture angryTexture; // Texture for the angry emotion
    public Texture neutralTexture; // Texture for the neutral emotion
    public Texture surpriseTexture; // Texture for the surprised emotion


    void Start()
    {
        StartCoroutine(GetEmotionFromPython());
    }

    // Coroutine to fetch subtitles from Python
    IEnumerator GetEmotionFromPython()
    {
        string url = "http://localhost:5000/receive_text"; // Replace with your Python server URL

        while (true)
        {
            using (var request = UnityEngine.Networking.UnityWebRequest.Get(url))
            {
                yield return request.SendWebRequest();

                if (request.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
                {
                    string emotionResult = request.downloadHandler.text;
                    if (emotionResult != null)
                    {
                        ChangeTexture(emotionResult);
                    }
                }
                else
                {
                    Debug.LogError("Error fetching subtitles: " + request.error);
                }
            }

            yield return new WaitForSeconds(1.0f); // Adjust the interval as needed
        }
    }

    // Function to change the texture based on the emotional input
    public void ChangeTexture(string emotion)
    {
        Texture newTexture = null;

        // Check the emotional input and assign the appropriate texture
        switch (emotion)
        {
            case "Happy":
                newTexture = happyTexture;
                break;
            case "Sad":
                newTexture = sadTexture;
                break;
            case "Angry":
                newTexture = angryTexture;
                break;
            case "Surprise":
                newTexture = surpriseTexture;
                break;
            case "Netural":
                newTexture = neutralTexture;
                break;
            default:
                Debug.LogWarning("Unrecognized emotion input");
                newTexture = neutralTexture;
                break;
        }

        // Change the model's texture
        if (newTexture != null && modelRenderer != null)
        {
            modelRenderer.material.mainTexture = newTexture;
        }
    }
}


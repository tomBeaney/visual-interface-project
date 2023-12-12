using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.Networking;

/*
[System.Serializable]
public class DATAPACKET //JSON packet containing string data 
{
    public string text;
}
*/

public class SubtitleController : MonoBehaviour
{
    public TMP_Text subtitleText; // Reference to the TextMeshPro component
    private string newText;

    void Start()
    {
        StartCoroutine(GetSubtitleFromPython());
    }

    // Coroutine to fetch subtitles from Python
    IEnumerator GetSubtitleFromPython()
    {
        string url = "http://localhost:5000/receive_text"; // Replace with your Python server URL

        while (true)
        {
            using (var request = UnityEngine.Networking.UnityWebRequest.Get(url))
            {
                yield return request.SendWebRequest();

                if (request.result == UnityEngine.Networking.UnityWebRequest.Result.Success)
                {
                    string subtitle = request.downloadHandler.text;
                    if (subtitle != null) {
                        UpdateSubtitle(subtitle);
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

    // Function to update the subtitle text
    public void UpdateSubtitle(string newText)
    {
        if (newText != null)
        {

            subtitleText.text = newText;
        }
    }
    //GPT CODE for json handling
    /*
    public string ParseTextFromJSON(string jsonString)
    {
        DATAPACKET data = JsonUtility.FromJson<DATAPACKET>(jsonString);
        if (data != null)
        {
            return data.text;
        }
        else
        {
            Debug.LogError("Failed to parse JSON or extract text.");
            return null;
        }
    */

}

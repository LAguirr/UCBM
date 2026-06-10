import React, { FC, useState, useRef, useEffect } from 'react';
import { useTranslation } from "react-i18next";

interface AudioRecorderProps {
    questionKey: string;
    onAudioRecorded: (questionKey: string, audioBlob: Blob) => void;
    onUploadAudio: (questionKey: string) => void;
}

const AudioRecorder: FC<AudioRecorderProps> = ({ questionKey, onAudioRecorded, onUploadAudio }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [audioURL, setAudioURL] = useState('');
    const [recordingTime, setRecordingTime] = useState(0);
    const [errorMsg, setErrorMsg] = useState<string | null>(null); // State to handle permission errors
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const { t } = useTranslation("guydeez");

    useEffect(() => {
        if (isRecording) {
            timerRef.current = setInterval(() => {
                setRecordingTime((prevTime) => prevTime + 1);
            }, 1000);
        } else if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }
        return () => {
            if (timerRef.current) {
                clearInterval(timerRef.current);
            }
        };
    }, [isRecording]);

    const startRecording = async () => {
        setErrorMsg(null); // Clear previous errors
        setRecordingTime(0);

        // 1. Check if the MediaDevices API is available (Catches HTTP / lack of support issues)
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            setErrorMsg("Microphone access is not supported. Please ensure you are on a secure connection (HTTPS).");
            return;
        }

        try {
            if (audioURL) {
                URL.revokeObjectURL(audioURL);
            }
            audioChunksRef.current = [];
            
            // 2. Trigger the permission prompt
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            mediaRecorderRef.current = new MediaRecorder(stream);
            mediaRecorderRef.current.ondataavailable = (event) => {
                audioChunksRef.current.push(event.data);
            };
            mediaRecorderRef.current.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);
                setAudioURL(audioUrl);
                onAudioRecorded(questionKey, audioBlob);

                // 3. Properly release the microphone hardware when stopped
                stream.getTracks().forEach(track => track.stop());
            };
            
            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (error: any) {
            console.error('Error accessing microphone:', error);
            setIsRecording(false);
            
            // 4. Handle specific permission errors gracefully in the UI
            if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                setErrorMsg("Microphone access was denied. Please allow permissions in your browser settings and try again.");
            } else if (error.name === 'NotFoundError') {
                setErrorMsg("No microphone was found on this device.");
            } else {
                setErrorMsg("An unexpected error occurred while trying to access the microphone.");
            }
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const handleReplaceRecording = () => {
        if (audioURL) {
            URL.revokeObjectURL(audioURL);
            setAudioURL('');
        }
        setRecordingTime(0);
        startRecording(); 
    };

    const formatTime = (seconds: number) => {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    };

    return (
        <div className="flex flex-col space-y-2">
            {/* Display validation/permission errors directly above the controls */}
            {errorMsg && (
                <div className="text-red-600 text-sm font-medium bg-red-50 p-2 rounded border border-red-200">
                    {errorMsg}
                </div>
            )}

            <div className="flex items-center space-x-4">
                {!audioURL && !isRecording && (
                    <button
                        onClick={startRecording}
                        className="bg-green-600 hover:bg-green-800 text-white font-bold py-2 px-4 rounded transition duration-300 ease-in-out"
                    >
                        {t('recruitment_audio_start_recording')}
                    </button>
                )}

                {isRecording && (
                    <div className="flex items-center space-x-2">
                        <button
                            onClick={stopRecording}
                            className="bg-red-600 hover:bg-red-800 text-white font-bold py-2 px-4 rounded transition duration-300 animate-pulse"
                        >
                            {t('recruitment_audio_stop_recording')}
                        </button>
                        <div className="flex items-center">
                            <div className="h-4 w-4 bg-red-600 rounded-full animate-ping"></div>
                            <span className="ml-2 text-gray-600">{formatTime(recordingTime)}</span>
                        </div>
                    </div>
                )}

                {audioURL && !isRecording && (
                    <div className="flex items-center space-x-4 flex-grow">
                        <fieldset className="border p-2 rounded w-full">
                            <legend className="font-semibold">{t('recruitment_audio_current_record_title')}</legend>
                            <div className="flex items-center justify-between">
                                <audio src={audioURL} controls className="outline-none" />
                                <span>{t('recruitment_audio_duration_label')}: {formatTime(recordingTime)}</span>
                                <button
                                    onClick={handleReplaceRecording}
                                    className="bg-blue-600 hover:bg-blue-800 text-white font-bold py-1 px-2 rounded transition duration-300 ease-in-out"
                                >
                                    {t('recruitment_audio_replace_recording')}
                                </button>
                            </div>
                        </fieldset>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AudioRecorder;
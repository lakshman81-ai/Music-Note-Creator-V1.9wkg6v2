
import React, { useEffect, useRef, useState } from 'react';
import * as OSMDModule from 'opensheetmusicdisplay';
import { NoteEvent, AlternativePitch } from '../types';

// Robust import handling for OSMD to support various build environments (ESM/CJS)
const getOSMDClass = () => {
    // @ts-ignore
    if (OSMDModule.OpenSheetMusicDisplay) return OSMDModule.OpenSheetMusicDisplay;
    // @ts-ignore
    if (OSMDModule.default?.OpenSheetMusicDisplay) return OSMDModule.default.OpenSheetMusicDisplay;
    // @ts-ignore
    if (typeof OSMDModule.default === 'function') return OSMDModule.default;
    return OSMDModule;
};

const OpenSheetMusicDisplay = getOSMDClass();

interface SheetMusicProps {
  musicXML?: string; // Content string
  transcribedNotes?: NoteEvent[];
  currentTime?: number;
  bpm?: number;
}

const SheetMusic: React.FC<SheetMusicProps> = ({ 
    musicXML, transcribedNotes = [], currentTime, bpm = 120
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const osmdRef = useRef<any>(null);
  const [isReady, setIsReady] = useState(false);

  // Floating Option Icon State
  const [activeAlternatives, setActiveAlternatives] = useState<AlternativePitch[]>([]);
  const [cursorPos, setCursorPos] = useState<{top: number, left: number} | null>(null);

  // Initialize OSMD
  useEffect(() => {
    if (!containerRef.current) return;

    // Cleanup previous instance if re-initializing
    if (osmdRef.current) {
        try {
            osmdRef.current.clear();
        } catch (e) { console.warn("OSMD Clear error", e); }
        osmdRef.current = null;
    }

    try {
        if (!OpenSheetMusicDisplay) {
            console.error("OpenSheetMusicDisplay library not loaded correctly.");
            return;
        }

        const osmd = new OpenSheetMusicDisplay(containerRef.current, {
            autoResize: true,
            backend: 'svg',
            drawingParameters: 'compacttight', 
            drawTitle: true,
            drawSubtitle: true,
            drawComposer: true,
            drawCredits: false,
            // Enable cursor options
            followCursor: true,
        });
        
        // Custom styling for the cursor
        osmd.setOptions({
            cursorsOptions: [{
                type: 1, // Vertical Line
                color: "#EF4444", // Red-500
                alpha: 0.8,
                width: 3,
            }],
        });

        osmdRef.current = osmd;
        setIsReady(true);
    } catch (e) {
        console.error("Failed to initialize OpenSheetMusicDisplay", e);
    }
  }, []); 

  // Load XML
  useEffect(() => {
      if (isReady && osmdRef.current && musicXML) {
          const loadScore = async () => {
              try {
                  await osmdRef.current.load(musicXML);
                  osmdRef.current.render();
                  if (osmdRef.current.cursor) {
                      osmdRef.current.cursor.show();
                      osmdRef.current.cursor.reset();
                  }
              } catch (e) {
                  console.error("OSMD Load Error:", e);
              }
          };
          loadScore();
      }
  }, [isReady, musicXML]);

  // Update Cursor based on currentTime
  useEffect(() => {
      if (isReady && osmdRef.current && osmdRef.current.cursor && musicXML && currentTime !== undefined) {
          try {
              // Convert seconds to beats
              // Assuming 4/4 and BPM provided (default 120 in generator)
              const secondsPerBeat = 60 / bpm;
              const currentBeat = currentTime / secondsPerBeat;
              
              const cursor = osmdRef.current.cursor;
              // Only move if we are not at the end
              if (!cursor.iterator.endReached) {
                  // Reset if we jumped back (naive)
                  if (currentTime < 0.2) cursor.reset();
                  
                  // Helper: Convert OSMD timestamp to seconds
                  // OSMD uses "Measures". We need "RealValue" which is beats (usually quarter notes).
                  const iteratorTime = cursor.iterator.currentTimeStamp.RealValue * 4; // MusicXML Measures to Beats (assuming 4/4)
                  
                  // If our visual cursor is behind real time, advance it
                  // We limit loop to avoid freezing
                  let steps = 0;
                  while (iteratorTime < currentBeat && !cursor.iterator.endReached && steps < 50) {
                      cursor.next();
                      steps++;
                  }

                  // --- Option Icon Logic ---
                  // 1. Get current visual cursor position
                  if (cursor.cursorElement) {
                      const rect = cursor.cursorElement.getBoundingClientRect();
                      const containerRect = containerRef.current?.getBoundingClientRect();

                      if (rect && containerRect) {
                          // Relative position inside the container
                          const top = rect.top - containerRect.top;
                          const left = rect.left - containerRect.left;
                          setCursorPos({ top, left: left + 15 }); // Offset slightly to the right
                      }
                  }

                  // 2. Check for alternatives
                  // Find the note that is currently playing or nearest
                  // We use currentTime
                  const activeNote = transcribedNotes.find(n =>
                      currentTime >= n.start_time && currentTime < (n.start_time + n.duration)
                  );

                  if (activeNote && activeNote.alternatives && activeNote.alternatives.length > 0) {
                      setActiveAlternatives(activeNote.alternatives);
                  } else {
                      setActiveAlternatives([]);
                  }
              }
          } catch(e) {
              // Ignore cursor errors during seek
          }
      }
  }, [currentTime, isReady, musicXML, bpm, transcribedNotes]);

  return (
    <div className="w-full h-full min-h-[400px] overflow-auto bg-white rounded-xl shadow-sm p-4 relative">
        <div ref={containerRef} className="w-full h-full relative" />

        {/* Floating Option Icon */}
        {cursorPos && activeAlternatives.length > 0 && (
            <div
                style={{ top: cursorPos.top, left: cursorPos.left }}
                className="absolute z-20 flex flex-col gap-1 animate-in fade-in slide-in-from-left-1 duration-200"
            >
                {activeAlternatives.map((alt, idx) => (
                    <div
                        key={idx}
                        className="flex items-center gap-1.5 bg-indigo-600 text-white text-[10px] font-bold px-2 py-1 rounded-full shadow-lg border border-white/20 hover:scale-105 transition-transform cursor-pointer"
                        title={`Alternative Source: ${alt.source} (Confidence: ${(alt.confidence * 100).toFixed(0)}%)`}
                    >
                        <span className="uppercase">{alt.source === 'hpss' ? 'H' : 'Alt'}</span>
                        <span className="opacity-80 font-mono">{alt.midi}</span>
                    </div>
                ))}
            </div>
        )}
    </div>
  );
};

export default SheetMusic;

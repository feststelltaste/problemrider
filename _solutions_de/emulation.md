---
title: Emulation
description: Nachbildung des Verhaltens einer fremden Plattform, sodass bestehende
  Software ohne Änderung läuft.
category:
- Operations
- Architecture
problems:
- obsolete-technologies
- technology-lock-in
- vendor-lock-in
- stagnant-architecture
- deployment-environment-inconsistencies
- legacy-skill-shortage
layout: solution
lang: de
en_slug: emulation
related_solutions:
- slug: containerization
  similarity: 0.8
- slug: platform-independent-data-storage
  similarity: 0.75
- slug: risk-analysis
  similarity: 0.75
- slug: abstraction-layers
  similarity: 0.75
- slug: automated-migration-tools
  similarity: 0.75
- slug: restore-points
  similarity: 0.75
---

## Description

Emulation bildet das Verhalten einer fremden Hardwareplattform oder eines Betriebssystems in Software nach, laufend auf moderner Infrastruktur, sodass eine Legacy-Anwendung unverändert weiterlaufen kann, selbst nachdem die ursprüngliche Hardware oder das OS, von dem sie abhing, veraltet oder unerreichbar geworden ist. Dies ist direkt relevant für Systeme, die Technology Lock-in um eingestellte Hardwareplattformen erleben, wo die Anwendungslogik selbst noch gültig und wertvoll sein kann — manchmal Jahrzehnte validierter, hart erkämpfter Domänenlogik repräsentierend — während das physische oder Plattform-Substrat, das sie benötigt, unter ihr verschwindet. Statt eine sofortige, hochriskante Neuschreibung dieser Logik unter Zeitdruck zu erzwingen, lässt Emulation den bestehenden Code genau so weiterlaufen, wie er immer lief, was der Organisation Zeit erkauft, eine echte Migration nach eigenem Zeitplan zu planen und ordentlich zu finanzieren, statt im Notfall eines Hardwareausfalls. Dies macht Emulation explizit zu einer Brückenstrategie statt zu einem Ziel: Sie kommt typischerweise mit einem Performance-Nachteil gegenüber nativer Ausführung, und emulierte Umgebungen können subtile Verhaltensunterschiede zur ursprünglichen Plattform beherbergen, die nur als seltene, schwer zu diagnostizierende Fehler auftauchen. Als dauerhafte Lösung statt bewusst zeitlich begrenzter behandelt, häuft sie auch ihr eigenes Risiko an, da das Emulations-Tooling selbst irgendwann unsupportet werden kann, was das ursprüngliche Veraltungsproblem effektiv nur eine Ebene tiefer verlagert, statt es zu lösen.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken zur Implementierung dieser Lösung im Kontext eines Legacy-Systems.

- Identifizieren Sie Legacy-Anwendungen, die von veralteter Hardware oder nicht mehr verfügbaren Betriebssystemen abhängen
- Bewerten Sie Emulationslösungen (Hardware-Emulatoren, OS-Kompatibilitätsschichten, Laufzeit-Emulatoren) für die Zielplattform
- Testen Sie die Legacy-Anwendung gründlich unter Emulation, um Verhaltenstreue zu verifizieren
- Nutzen Sie Emulation als Brückenstrategie, während eine echte Migration oder Neuschreibung geplant wird
- Dokumentieren Sie das Emulations-Setup, sodass es reproduziert werden kann, falls die Emulationsumgebung neu gebaut werden muss
- Überwachen Sie die Performance unter Emulation und etablieren Sie akzeptable Performance-Schwellenwerte

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie aufgeben.

**Vorteile:**
- Verlängert die Lebensdauer von Legacy-Software ohne jegliche Codeänderungen
- Erkauft Zeit für die Planung und Ausführung einer echten Migrationsstrategie
- Kann geschäftskritische Funktionalität bewahren, die teuer neu zu schreiben wäre

**Kosten und Risiken:**
- Emulation verursacht typischerweise Performance-Overhead im Vergleich zu nativer Ausführung
- Emulierte Umgebungen könnten subtile Verhaltensunterschiede haben, die als seltene Fehler auftauchen
- Sich unbegrenzt auf Emulation zu verlassen erhöht technische Schulden und operatives Risiko
- Emulationswerkzeuge selbst könnten unsupportet oder veraltet werden

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Verteidigungsunternehmen betrieb missionskritische Simulationssoftware auf einer Solaris-SPARC-Plattform, die sich dem Ende des Herstellersupports näherte. Statt die Simulation neu zu schreiben, die Jahrzehnte validierter Physikmodelle enthielt, deployte das Team sie unter einem SPARC-Emulator auf moderner x86-Hardware. Während die Performance 30 Prozent langsamer war, waren die Simulationsergebnisse identisch. Dies erkaufte der Organisation drei Jahre, um eine echte Migration zu einer modernen Plattform zu planen und zu finanzieren, während ununterbrochener Zugang zur Simulation aufrechterhalten wurde.

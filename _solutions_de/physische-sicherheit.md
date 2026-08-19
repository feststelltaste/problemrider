---
title: Physische Sicherheit
description: Zugangs- und Zutrittsschutz für IT-Infrastruktur durch bauliche
  und organisatorische Maßnahmen.
category:
- Security
- Operations
problems:
- data-protection-risk
- system-outages
- regulatory-compliance-drift
- poor-system-environment
- monitoring-gaps
- secret-management-problems
layout: solution
lang: de
en_slug: physical-security
related_solutions:
- slug: encryption
  similarity: 0.7
- slug: logging-and-monitoring
  similarity: 0.7
- slug: authentication
  similarity: 0.7
- slug: honeypots
  similarity: 0.7
- slug: endpoint-detection-and-response
  similarity: 0.7
- slug: malware-protection
  similarity: 0.7
---

## Description

Physische Sicherheit schützt die tatsächliche Hardware, auf der ein Legacy-System läuft — Serverräume, Netzwerkschränke, Wechseldatenträger, Arbeitsplätze mit direktem Systemzugang — vor unautorisiertem Zugang, Diebstahl, Manipulation und Umweltschäden, durch bauliche und organisatorische Kontrollen wie Zutrittskarten, Umgebungsüberwachung und Besuchermanagement. Legacy-Infrastruktur ist hier überproportional exponiert, weil sie oft moderne Anlagenstandards vordatiert und über Jahre des Betriebs gemeinsame Schlüssel, undokumentierten Zugang und unüberwachte Eingangspunkte angesammelt hat, wofür keine logische Sicherheitskontrolle kompensieren kann: physischer Zugang zur Hardware besiegt Software-Schutzmaßnahmen vollständig. Die Etablierung individuell nachverfolgter Zugangsberechtigungen, Umgebungsüberwachung und sicherer Vernichtungsverfahren für ausgemusterte Datenträger schließt eine Angriffsfläche, die leicht zu übersehen ist, gerade weil sie kein Code- oder Netzwerkproblem ist, erfordert aber Investition in Anlagen, die mit anderen Sicherheitsprioritäten um Budget konkurriert.

## How to Apply ◆

> Legacy-Systeme laufen oft auf physischer Hardware in Serverräumen mit veralteten oder unzureichenden physischen Zugangskontrollen. Physische Sicherheit schützt IT-Infrastruktur vor unautorisiertem physischem Zugang, Diebstahl, Manipulation und Umweltgefahren.

- Prüfen Sie den physischen Zugang zu allen Orten, an denen sich Legacy-System-Hardware befindet: Serverräume, Netzwerkschränke, Bandspeicherbereiche und alle Arbeitsplätze mit direktem Zugang zu Legacy-System-Schnittstellen. Identifizieren Sie, wer derzeit Zugang hat und ob dieser Zugang gerechtfertigt ist.
- Implementieren Sie Zugangskontrollmechanismen (Kartenleser, biometrische Scanner, abgeschlossene Schränke) für alle Bereiche mit Legacy-System-Hardware. Ersetzen Sie gemeinsame Schlüssel und Türcodes durch individuell nachverfolgte Zugangsberechtigungen.
- Setzen Sie Umgebungsüberwachung für Serverräume ein: Temperatursensoren, Feuchtigkeitssensoren, Wasserleck-Erkennung und Rauchmelder. Legacy-Hardware kann engere Umgebungstoleranzen haben als moderne Geräte.
- Implementieren Sie Videoüberwachung und Zugangsprotokollierung für sensible Bereiche. Physische Zugangsprotokolle sollten aufbewahrt und periodisch überprüft werden, und Zugangsereignisse sollten mit autorisierten Arbeitsaufträgen korrelieren.
- Sichern Sie Wechseldatenträger und tragbare Speicher. Legacy-Systeme nutzen oft USB-Laufwerke, Bänder oder Wechseldatenträger für Datenübertragung und Backup — diese müssen verschlüsselt, nachverfolgt und sicher aufbewahrt werden, wenn nicht in Gebrauch.
- Implementieren Sie Besuchermanagement-Verfahren für Bereiche mit Legacy-System-Infrastruktur: Begleitpflicht, temporäre Zugangsausweise und Ein-/Ausgangsprotokolle.
- Planen Sie die physische Sicherheit von Legacy-Hardware während Umzügen, Außerbetriebnahme und Entsorgung. Festplatten, Bänder und andere Datenträger mit sensiblen Daten müssen sicher gelöscht oder zerstört werden, wenn Hardware ausgemustert wird.

## Tradeoffs ⇄

> Physische Sicherheit verhindert unautorisierten physischen Zugang zur Infrastruktur und schützt vor Bedrohungen, die logische Kontrollen nicht adressieren können, erfordert aber Investition in Anlagen, Ausrüstung und laufende Betriebsverfahren.

**Vorteile:**

- Verhindert Datendiebstahl, Hardware-Manipulation und unautorisierten Zugang, die alle logischen Sicherheitskontrollen umgehen — physischer Zugang zur Hardware besiegt die meisten Software-Schutzmaßnahmen.
- Schützt vor Umweltbedrohungen (Feuer, Überschwemmung, Stromausfall), die Legacy-Hardware und die darauf befindlichen Daten zerstören können.
- Unterstützt Compliance mit Sicherheitsstandards (ISO 27001, PCI DSS, HIPAA), die dokumentierte physische Zugangskontrollen vorschreiben.
- Liefert einen Prüfpfad physischen Zugangs für Untersuchungs- und Compliance-Zwecke.

**Kosten und Risiken:**

- Verbesserungen der physischen Sicherheit (Zugangskontrollsysteme, Überwachung, Umgebungsüberwachung) erfordern Kapitalinvestition in Anlageninfrastruktur.
- Übermäßig restriktiver physischer Zugang kann legitime Wartungs- und Notfallreaktionsaktivitäten für Legacy-Systeme verzögern, die manuellen Eingriff erfordern.
- Legacy-Systeme an entfernten Standorten (Filialen, Fabrikhallen, Kundenstandorte) können schwer auf denselben Standard wie ein Rechenzentrum physisch abzusichern sein.
- Die Außerbetriebnahme von Legacy-Hardware mit sensiblen Daten erfordert sichere Vernichtungsverfahren, die Kosten und Komplexität hinzufügen.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie physische Sicherheit Legacy-System-Infrastruktur schützt.

Ein Legacy-Datenbankserver mit 10 Jahren Kundenfinanzdaten steht in einem Serverraum, der nur durch ein Zahlenschloss gesichert ist. Die Kombination wurde seit 5 Jahren nicht geändert und ist mindestens 30 aktuellen und ehemaligen Mitarbeitern bekannt. Während eines Einbruchs außerhalb der Geschäftszeiten wird eine Festplatte aus dem Server entfernt, und der Diebstahl wird erst am nächsten Geschäftstag entdeckt, als das System nicht mehr bootet. Das Team implementiert kartenbasierte Zugangskontrolle mit individuellen Zugangsprotokollen, fügt 24/7-Videoüberwachung mit 90-tägiger Aufbewahrung hinzu, setzt Manipulationserkennungssensoren an Servergehäusen ein und aktiviert Festplattenverschlüsselung, sodass gestohlene Laufwerke ohne den Verschlüsselungsschlüssel unlesbar sind. Der Zugang wird auf 5 autorisierte Mitarbeiter beschränkt, und Zugangsüberprüfungen werden monatlich durchgeführt. Das Zahlenschloss wird durch einen Kartenleser ersetzt, der für jeden Zutritt ein prüfbares Zugangsprotokoll erzeugt.

Ein Unternehmen betreibt Legacy-Fertigungssteuerungssysteme auf über eine Fabrikhalle verteilten Arbeitsplätzen. Diese Arbeitsplätze haben aktivierte USB-Ports für Datenübertragung und sind physisch für das gesamte Fabrikpersonal zugänglich. Eine Sicherheitsbewertung offenbart, dass jeder ein USB-Gerät mit Malware einstecken oder Daten aus dem Fertigungssystem kopieren kann. Das Team implementiert USB-Port-Sperren an allen Legacy-Arbeitsplätzen, installiert verschließbare Gehäuse, die den Zugang zum Computergehäuse verhindern, und setzt einen dedizierten Kiosk für genehmigte Datenübertragungen mit Malware-Scanning ein. Vierteljährlich durchgeführte physische Zugangsprüfungen verifizieren, dass die Kontrollen intakt bleiben und keine unautorisierten Modifikationen an den Legacy-Arbeitsplätzen vorgenommen wurden.

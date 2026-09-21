---
title: Sensibilisierung der Nutzer
description: Sensibilisierung und Schulung von Mitarbeitern und Nutzern zu
  Sicherheitsthemen.
category:
- Security
- Culture
problems:
- knowledge-gaps
- inadequate-onboarding
- implicit-knowledge
- fear-of-change
- resistance-to-change
- workaround-culture
- password-security-weaknesses
layout: solution
lang: de
en_slug: raising-user-awareness
related_solutions:
- slug: security-training
  similarity: 0.85
- slug: security-policies-for-users
  similarity: 0.8
- slug: security-community
  similarity: 0.8
- slug: security-certification
  similarity: 0.75
- slug: security-incident-handling
  similarity: 0.75
- slug: vulnerability-scans
  similarity: 0.75
---

## Description

Sensibilisierung der Nutzer ist eine Reihe fortlaufender Bildungsaktivitäten — Schulungssitzungen, simulierte Phishing-Kampagnen, rollenspezifische Module, Sicherheitsbulletins —, die darauf abzielen, eine menschliche Verteidigungsschicht gegen Bedrohungen wie Social Engineering und Credential-Diebstahl aufzubauen, die technische Kontrollen allein nicht vollständig adressieren können, da viele Angriffe speziell darauf ausgelegt sind, das Urteilsvermögen einer Person auszunutzen statt den Code eines Systems. Auf Legacy-Umgebungen angewendet, zielt sie direkt auf eine häufige und spezifische Schwäche: Legacy-Systeme werden überproportional wahrscheinlich mit gemeinsamen Konten, schwachen Passwörtern und anderen informellen Zugangspraktiken betrieben, die sich über eine lange Betriebsgeschichte angesammelt haben, gerade weil die Menschen, die sie nutzten, nie einen strukturierten Grund erhielten, diese Gewohnheiten zu ändern. Awareness-Programme fungieren auch als ungeplanter Entdeckungsmechanismus — Mitarbeiter zu bitten, kritisch über ihre eigenen Zugänge und Anmeldedaten nachzudenken, tendiert dazu, undokumentierte gemeinsame Konten und andere Artefakte derselben informellen Geschichte offenzulegen, Befunde, die dann ein breiteres Zugangs-Review rechtfertigen. Der Ansatz ist ergänzend statt eines Ersatzes für technische Härtung, da verbesserte Sensibilisierung die Wahrscheinlichkeit und Auswirkung von Social Engineering reduziert, aber allein nichts tut, um eine technische Schwachstelle anderswo im Legacy-Stack zu schließen. Ihre Hauptkosten sind die Notwendigkeit kontinuierlicher Inhaltsauffrischung, während sich Bedrohungen weiterentwickeln, das Reputationsrisiko, dass übermäßig aggressive simulierte Angriffe Vertrauen beschädigen, und die generelle Schwierigkeit, einen klaren Return on Investment für ein Programm zu beweisen, dessen Erfolg an Vorfällen gemessen wird, die nicht passiert sind.

## How to Apply ◆

> Konkrete Schritte, Ansätze oder Praktiken, um diese Lösung im Kontext eines Legacy-Systems umzusetzen.

- Führen Sie regelmäßige Sicherheitsbewusstseins-Schulungssitzungen durch, die häufige Bedrohungen wie Phishing, Social Engineering und Credential-Diebstahl abdecken
- Erstellen Sie rollenspezifische Schulungsmodule, die die für jede Nutzergruppe relevanten Sicherheitsbedenken adressieren
- Führen Sie simulierte Phishing-Kampagnen durch, um Sensibilisierungsgrade zu messen und Bereiche zu identifizieren, die Verbesserung brauchen
- Etablieren Sie klare Meldekanäle für Nutzer, um verdächtige Aktivitäten oder potenzielle Sicherheitsvorfälle zu markieren
- Integrieren Sie Sicherheitssensibilisierung in Onboarding-Programme für neue Mitarbeiter und Auftragnehmer
- Verteilen Sie regelmäßige Sicherheitsbulletins, die aktuelle Bedrohungen und bewährte Praktiken hervorheben
- Gamifizieren Sie Sicherheitssensibilisierung durch Quizze, Wettbewerbe und Anerkennungsprogramme

## Tradeoffs ⇄

> Was Sie durch die Anwendung dieser Lösung gewinnen und was Sie dafür aufgeben.

**Vorteile:**
- Reduziert die Wahrscheinlichkeit erfolgreicher Social-Engineering-Angriffe
- Schafft eine menschliche Verteidigungsschicht, die technische Sicherheitskontrollen ergänzt
- Verbessert Geschwindigkeit und Qualität der Vorfallmeldung
- Baut eine sicherheitsbewusste Kultur auf, die über einzelne Schulungsereignisse hinaus fortbesteht

**Kosten und Risiken:**
- Schulungsprogramme erfordern laufende Investition und regelmäßige Inhaltsaktualisierungen
- Übermäßig aggressive simulierte Angriffe können Mitarbeitervertrauen und -moral beschädigen
- Sensibilisierung allein verhindert nicht alle Angriffe; technische Kontrollen bleiben essenziell
- Die Messung des ROI von Awareness-Programmen ist von Natur aus schwierig

## How It Could Be

> Konkrete Beispiele oder Szenarien aus Legacy-System-Kontexten, die diese Lösung in der Praxis veranschaulichen.

Ein Fertigungsunternehmen, das Legacy-ERP-Systeme betrieb, erlebte wiederholte Kompromittierung von Anmeldedaten, weil Mitarbeiter einfache Passwörter und gemeinsame Konten nutzten. Das Sicherheitsteam führte vierteljährliche Sensibilisierungssitzungen kombiniert mit monatlichen simulierten Phishing-E-Mails ein. Innerhalb von sechs Monaten sanken die Klickraten bei Phishing von 32 % auf 8 %, und Mitarbeiter begannen, verdächtige E-Mails proaktiv zu melden. Die Initiative führte außerdem zur Entdeckung mehrerer gemeinsamer Servicekonten im Legacy-System, die nie dokumentiert worden waren, was ein breiteres Zugangs-Review anstieß.

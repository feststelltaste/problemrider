---
title: Prototypen
description: Frühzeitige Validierung von Eignung und Nutzbarkeit durch
  fachliche Prototypen.
category:
- Requirements
- Process
problems:
- implementation-rework
- misaligned-deliverables
- requirements-ambiguity
- poor-user-experience-ux-design
- customer-dissatisfaction
- fear-of-change
- modernization-strategy-paralysis
- assumption-based-development
- decision-paralysis
- inability-to-innovate
- premature-technology-introduction
- reduced-innovation
- decision-avoidance
layout: solution
lang: de
en_slug: prototypes
related_solutions:
- slug: prototyping
  similarity: 0.95
- slug: wireframing
  similarity: 0.8
- slug: user-stories
  similarity: 0.75
- slug: personas
  similarity: 0.75
- slug: on-site-customer
  similarity: 0.75
- slug: usability-tests
  similarity: 0.75
---

## Description

Fachliche Prototypen sind bewusst unvollständige, wegwerfbare Darstellungen eines vorgeschlagenen Systems — Wireframes, klickbare Mockups oder eng begrenzte funktionsfähige Software —, speziell gebaut, um Stakeholdern und Endnutzern zu erlauben, ein vorgeschlagenes Design zu sehen und mit ihm zu interagieren, bevor irgendeine Festlegung auf die vollständige Implementierung erfolgt. Ihr definierendes Merkmal ist, dass sie Wegwerfartefakte sind: Das Ziel ist validiertes Lernen darüber, ob eine Designrichtung tatsächlich für ihre Nutzer funktioniert, nicht eine Codebasis, die in Produktion erweitert wird. Dies zählt akut in der Legacy-Modernisierung, weil die Menschen, die ein Legacy-System jahrelang genutzt haben, starke, oft undokumentierte Erwartungen daran angesammelt haben, wie sich Arbeitsabläufe anfühlen sollten, und diese Erwartungen überleben selten unversehrt ein schriftliches Anforderungsdokument — ein Sachbearbeiter, der fünfzehn Jahre lang zwischen mehreren offenen Panels in einer Legacy-Oberfläche multitaskt hat, wird in einer fünfminütigen Prototyp-Sitzung sofort einen linearen Ersatz-Workflow bemerken, den ein Anforderungsreview nie markiert hätte. Prototypen der tatsächlichen Population von Legacy-System-Nutzern vorzulegen, statt nur Projekt-Stakeholdern, legt genau diese Art von Workflow-Fehlanpassung offen, während sie noch günstig zu ändern ist, Monate bevor sie sonst in der Abnahmeprüfung entdeckt würde. Das zentrale Risiko, das Prototypen von gewöhnlicher früher Entwicklung unterscheidet, ist Umfangsverwirrung: Stakeholder können einen funktionsfähig aussehenden Prototyp leicht für nahezu fertige Software halten, und übereilter Prototyp-Code, der nicht klar als wegwerfbar begrenzt ist, hat die Tendenz, in Produktion ausgeliefert zu werden und dabei seine Abkürzungen mitzubringen.

## How to Apply ◆

> Fachliche Prototypen in der Legacy-Modernisierung lassen Stakeholder vorgeschlagene Ersatzsysteme sehen und mit ihnen interagieren, bevor sie sich auf die vollständige Implementierung festlegen, was das Risiko reduziert, das Falsche zu bauen.

- Bauen Sie Low-Fidelity-Prototypen (Wireframes, klickbare Mockups) der Schlüsselarbeitsabläufe des Ersatzsystems und präsentieren Sie sie tatsächlichen Nutzern des Legacy-Systems für Feedback.
- Konzentrieren Sie Prototypen auf die Arbeitsabläufe, bei denen das Legacy-System am schmerzhaftesten ist oder wo das Ersatzdesign am meisten vom bestehenden Verhalten abweicht, da dies die risikoreichsten Bereiche für Nutzerablehnung sind.
- Nutzen Sie Prototypen, um zu validieren, dass kritisches Legacy-System-Verhalten erhalten bleibt — Nutzer haben oft starke Erwartungen, geformt durch Jahre der Nutzung des alten Systems.
- Iterieren Sie Prototypen schnell basierend auf Nutzer-Feedback und behandeln Sie jede Version als Lernwerkzeug statt als Festlegung auf ein spezifisches Design.
- Erstellen Sie High-Fidelity-Prototypen für die kritischsten oder umstrittensten Features, um Stakeholdern Vertrauen zu geben, bevor Entwicklungsressourcen gebunden werden.
- Präsentieren Sie Prototypen verschiedenen Nutzergruppen separat, um abweichende Bedürfnisse und Arbeitsabläufe zu erfassen.

## Tradeoffs ⇄

> Prototypen beschleunigen die Ausrichtung zwischen Stakeholdern und Entwicklern, müssen aber klar als wegwerfbare Artefakte positioniert werden, um Umfangsverwirrung zu vermeiden.

**Vorteile:**

- Reduziert kostspieligen Nacharbeitsaufwand, indem Usability-Probleme und Anforderungslücken vor Beginn der vollständigen Entwicklung identifiziert werden.
- Hilft Stakeholdern, die Schwierigkeiten haben, Anforderungen abstrakt zu artikulieren, konkretes Feedback zu geben, wenn sie eine vorgeschlagene Lösung sehen und mit ihr interagieren können.
- Baut Stakeholder-Vertrauen in die Modernisierungsanstrengung auf, indem Fortschritt früh sichtbar gemacht wird.
- Legt versteckte Anforderungen und unausgesprochene Annahmen über Legacy-System-Verhalten offen, die schriftliche Spezifikationen übersehen.

**Kosten und Risiken:**

- Stakeholder könnten einen Prototyp für ein nahezu fertiges Produkt halten und den verbleibenden Entwicklungsaufwand unterschätzen.
- In Produktion überstürzter Prototyp-Code erzeugt technische Schulden vom Beginn der Modernisierungsanstrengung an.
- Der Bau von Prototypen erfordert Designfähigkeiten und Tooling, die legacy-fokussierte Teams möglicherweise nicht ohne Weiteres zur Verfügung haben.
- Übermäßiges Prototyping kann die tatsächliche Entwicklung verzögern, wenn das Team durch zu viele Iterationen zykelt, ohne sich auf eine Implementierung festzulegen.

## How It Could Be

> Das folgende Szenario veranschaulicht, wie Prototypen kostspielige Fehler während des Legacy-Ersatzes verhindern.

Ein Versicherungsunternehmen ersetzte ein Legacy-Policenverwaltungssystem, das Vermittler 15 Jahre lang genutzt hatten. Das Entwicklungsteam baute einen ersten Prototyp basierend auf Anforderungsdokumenten und präsentierte ihn einer Gruppe von Vermittlern. Innerhalb von 30 Minuten identifizierten die Vermittler, dass der lineare Workflow des Prototyps für Policenänderungen die benötigte Zeit im Vergleich zum Multi-Panel-Ansatz des Legacy-Systems, der ihnen erlaubte, mehrere Abschnitte gleichzeitig anzusehen und zu bearbeiten, verdreifachen würde. Ohne den Prototyp wäre diese grundlegende Workflow-Fehlanpassung erst Monate später während der Abnahmeprüfung entdeckt worden. Das Team gestaltete die Oberfläche um ein Tab-Layout um, das die Fähigkeit zur Mehrfach-Abschnitts-Bearbeitung bewahrte, validierte es mit einer zweiten Prototyp-Runde und ging dann mit hohem Vertrauen zur Implementierung über.

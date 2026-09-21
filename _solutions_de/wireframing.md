---
title: Wireframing
description: Erstellung vorläufiger visueller Darstellungen als
  Diskussionsgrundlage.
category:
- Requirements
- Process
quality_tactics_url: https://qualitytactics.de/en/usability/wireframing/
problems:
- poor-user-experience-ux-design
- implementation-starts-without-design
- requirements-ambiguity
- stakeholder-developer-communication-gap
- misaligned-deliverables
- implementation-rework
- feature-gaps
- user-frustration
layout: solution
lang: de
en_slug: wireframing
related_solutions:
- slug: prototypes
  similarity: 0.8
- slug: prototyping
  similarity: 0.8
- slug: story-mapping
  similarity: 0.75
- slug: user-stories
  similarity: 0.75
- slug: user-centered-design
  similarity: 0.75
- slug: on-site-customer
  similarity: 0.7
---

## Description

Wireframing produziert ein kostengünstiges, wenig ausgereiftes visuelles Layout eines Bildschirms — auf Papier skizziert oder in einem einfachen Werkzeug —, auf das Stakeholder und Nutzer reagieren und das sie verfeinern können, bevor irgendein Code geschrieben wird, statt Implementierung auf nichts weiter als dem eigenen Verständnis des Teams von einer Anforderung beginnen zu lassen. Legacy-Modernisierungsarbeit neigt besonders dazu, diesen Schritt zu überspringen, da oft Druck besteht, einfach mit dem Bauen zu beginnen, angesichts dessen, wie viel bereits über das alte System bekannt ist, aber genau diese Abkürzung ist es, die dazu führt, dass ein Team mehrere Sprints an einem Bildschirm verbringt, der sich als überhaupt nicht zum tatsächlichen Workflow passend herausstellt. Am Wireframe selbst zu iterieren kostet fast nichts im Vergleich zur Überarbeitung implementierten Codes, was die gesamte Rechtfertigung dafür ist, ein oder zwei Tage hierfür aufzuwenden, bevor die Entwicklung beginnt, statt nachdem die Diskrepanz auf teure Weise entdeckt wurde.

## How to Apply ◆

> Legacy-Systemmodernisierung beginnt oft mit dem Codieren, bevor das Team ein gemeinsames Verständnis davon hat, wie die verbesserte Schnittstelle aussehen sollte. Wireframing schafft kostengünstige visuelle Darstellungen, die Stakeholder ausrichten, bevor die Entwicklung beginnt.

- Erstellen Sie wenig ausgereifte Wireframes mit einfachen Werkzeugen wie Stift und Papier, Balsamiq oder Figma, bevor irgendeine Entwicklungsarbeit an Schnittstellenänderungen beginnt. Das Ziel ist, Layout- und Interaktionsoptionen zu erkunden, nicht polierte Designs zu produzieren.
- Nutzen Sie Wireframes, um Diskussionen mit Stakeholdern und Nutzern darüber zu erleichtern, welche Informationen und Aktionen auf jedem Bildschirm erscheinen sollten, wie sie organisiert sein sollten und wie der Workflow zwischen Bildschirmen aussehen sollte.
- Erstellen Sie Wireframes sowohl für den aktuellen Zustand als auch den vorgeschlagenen zukünftigen Zustand, sodass Stakeholder sehen können, was sich ändern wird, und fundiertes Feedback geben können.
- Testen Sie Wireframes mit repräsentativen Nutzern durch Papier-Prototyp-Testing oder klickbare Prototyp-Durchgänge, um Usability-Probleme zu identifizieren, bevor irgendein Code geschrieben wird.
- Iterieren Sie schnell über Wireframes basierend auf Feedback. Die Kosten, ein Wireframe zu ändern, sind vernachlässigbar im Vergleich zu den Kosten, implementierten Code zu ändern.
- Nutzen Sie Wireframes, um Schnittstellenentscheidungen zu dokumentieren und als Spezifikation für Entwickler zu dienen, was Mehrdeutigkeit darüber reduziert, was gebaut werden soll.

## Tradeoffs ⇄

> Wireframing verhindert teure Nacharbeit, indem Designentscheidungen früh validiert werden, fügt aber einen Designschritt hinzu, gegen den sich Teams sträuben könnten, die daran gewöhnt sind, direkt zu codieren.

**Vorteile:**

- Erfasst Designprobleme und Anforderungsmissverständnisse, bevor Code geschrieben wird, wenn Änderungen am günstigsten und einfachsten sind.
- Schafft eine gemeinsame visuelle Sprache zwischen Entwicklern, Stakeholdern und Nutzern, was die Kommunikationslücke reduziert, die zu fehlausgerichteten Liefergütern führt.
- Reduziert Implementierungs-Nacharbeit, verursacht durch den Bau des Falschen, weil das Team kein klares Bild des Ziels hatte.
- Ermöglicht schnelle Erkundung mehrerer Designalternativen zu geringen Kosten, bevor man sich auf einen Ansatz festlegt.

**Kosten und Risiken:**

- Das Hinzufügen eines Wireframing-Schritts verlängert den Zeitplan, bevor die Entwicklung beginnt, was in Organisationen, die Geschwindigkeit priorisieren, als Verzögerung wahrgenommen werden kann.
- Zu polierte Wireframes können unrealistische Erwartungen über die endgültige visuelle Qualität schaffen, besonders wenn der Legacy-Technologie-Stack begrenzt, was erreichbar ist.
- Stakeholder könnten sich auf Wireframe-Ästhetik statt auf Struktur und Interaktionsfluss konzentrieren, was Diskussionen über das Wesentliche entgleisen lässt.
- Wireframes können schnell veraltet werden, wenn sie nicht gepflegt werden, während sich das Design während der Implementierung weiterentwickelt, und werden zu irreführenden Artefakten.

## How It Could Be

> Viele Legacy-Modernisierungsprojekte scheitern, weil das Team eine Schnittstelle baut, die niemand vor der Implementierung überprüft oder validiert hat.

Ein Legacy-Abrechnungssystem wird modernisiert, und das Entwicklungsteam plant, den Rechnungserstellungsbildschirm neu zu bauen. Ohne Wireframes baut das Team den neuen Bildschirm basierend auf seinem Verständnis der Anforderungen und verbringt drei Sprints mit der Implementierung. Als Stakeholder das Ergebnis überprüfen, entdecken sie, dass der Workflow nicht dem Prozess des Buchhaltungsteams entspricht: Das Team baute ein Einzelschritt-Formular, während die Buchhalter einen mehrstufigen Workflow mit Genehmigungskontrollpunkten brauchen. Der gesamte Bildschirm muss substanziell überarbeitet werden. Für das nächste Modul übernimmt das Team Wireframing. Sie verbringen zwei Tage damit, Wireframes des vorgeschlagenen Zahlungsabgleichsbildschirms zu erstellen, überprüfen sie mit dem Buchhaltungsteam und iterieren durch drei Versionen basierend auf Feedback. Als die Entwicklung beginnt, hat das Team ein validiertes Design, das dem tatsächlichen Workflow entspricht. Die Implementierung verläuft reibungslos ohne Nacharbeit, und das Buchhaltungsteam fühlt Ownership des Designs, weil sein Input es von Anfang an geformt hat.

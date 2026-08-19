---
title: Definition von Missbrauchsfällen
description: Beschreibung unerwünschter Nutzungsfälle aus der Perspektive von Angreifern.
category:
- Security
- Requirements
problems:
- authentication-bypass-vulnerabilities
- authorization-flaws
- cross-site-scripting-vulnerabilities
- sql-injection-vulnerabilities
- buffer-overflow-vulnerabilities
- quality-blind-spots
- inadequate-requirements-gathering
- data-protection-risk
layout: solution
lang: de
en_slug: abuse-case-definition
related_solutions:
- slug: authentication
  similarity: 0.75
- slug: negative-testing
  similarity: 0.7
- slug: penetration-tests
  similarity: 0.7
- slug: authorization
  similarity: 0.7
- slug: logging-and-monitoring
  similarity: 0.7
- slug: honeypots
  similarity: 0.7
---

## Description

Die Definition von Missbrauchsfällen ist eine Anforderungsanalysetechnik, die systematisch dokumentiert, wie die Funktionalität eines Systems missbraucht oder angegriffen werden könnte, und für jeden legitimen Anwendungsfall, um den das System ursprünglich designt wurde, einen begleitenden Missbrauchsfall produziert. Wo ein Anwendungsfall beschreibt, was ein autorisierter Nutzer zu erreichen versucht, beschreibt sein Missbrauchsfall, was ein Angreifer durch Ausnutzung derselben Funktionalität erreichen könnte, zusammen mit dem Akteur, dem Angriffsvektor, den Voraussetzungen und der resultierenden Auswirkung. Legacy-Systeme sind dieser Lücke besonders ausgesetzt, weil ihre ursprünglichen Anforderungen typischerweise geschrieben wurden, als das System hinter einer Firewall oder einem internen Netzwerk operierte, und Sicherheitsbedrohungen weder antizipiert noch als erstrangige Anforderungen dokumentiert wurden; über die Zeit ist die tatsächliche Exposition des Systems gewachsen, während seine Anforderungen nicht überprüft wurden. Die rückwirkende Definition von Missbrauchsfällen verwandelt implizite, unausgesprochene Sicherheitsannahmen in explizite und testbare Aussagen, was dem Team einen strukturierten Weg gibt, zu entscheiden, welche Bedrohungen am wichtigsten sind, und sie direkt in konkrete Sicherheitstestfälle zu übersetzen. Weil die Übung sowohl tiefes Wissen über die Internas des Legacy-Systems als auch aktuelles Wissen über Angreifertechniken erfordert, funktioniert sie am besten als kollaborativer Workshop zwischen Entwicklern und Sicherheitsspezialisten, statt als Aufgabe, die eine Gruppe allein abschließen kann. Der resultierende Missbrauchsfall-Katalog dient außerdem als dauerhafte Dokumentation, die nachfolgende Sicherheitsinvestitionen gegenüber Stakeholdern rechtfertigt, die das Risiko in Funktionalität, die „schon immer gut funktioniert hat", sonst vielleicht nicht sehen würden.

## How to Apply ◆

> Legacy-Systeme werden typischerweise nur mit legitimen Anwendungsfällen im Sinn gebaut, wobei Sicherheitsüberlegungen zu einem nachträglichen Gedanken werden. Die Definition von Missbrauchsfällen identifiziert systematisch, wie Angreifer das System missbrauchen können, und verwandelt implizite Sicherheitsannahmen in explizite, testbare Anforderungen.

- Erstellen Sie für jeden bestehenden Anwendungsfall im Legacy-System einen entsprechenden Missbrauchsfall, der beschreibt, wie ein Angreifer diese Funktionalität ausnutzen oder missbrauchen könnte. Wenn der Anwendungsfall zum Beispiel „Nutzer loggt sich ein" ist, ist der Missbrauchsfall „Angreifer erzwingt Login-Zugangsdaten per Brute-Force" oder „Angreifer umgeht Authentifizierung via Session Fixation."
- Beziehen Sie sowohl Entwickler als auch Sicherheitsspezialisten in Missbrauchsfall-Workshops ein. Entwickler verstehen die Internas des Systems und wissen, wo Abkürzungen genommen wurden, während Sicherheitsspezialisten Wissen über häufige Angriffsmuster und Ausnutzungstechniken einbringen.
- Priorisieren Sie die Missbrauchsfallanalyse für die sicherheitssensibelsten Teile des Legacy-Systems: Authentifizierung, Autorisierung, Zahlungsverarbeitung, Handhabung personenbezogener Daten und administrative Funktionen.
- Dokumentieren Sie jeden Missbrauchsfall mit einem Bedrohungsakteur-Profil (wer würde dies versuchen), dem Angriffsvektor (wie würden sie es tun), den Voraussetzungen (welchen Zugriff oder welches Wissen benötigen sie) und der Auswirkung (welcher Schaden würde resultieren). Dieses strukturierte Format macht Missbrauchsfälle sowohl für Testen als auch Behebung umsetzbar.
- Nutzen Sie das STRIDE-Framework (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege) als Checkliste, um umfassende Abdeckung über verschiedene Bedrohungskategorien sicherzustellen.
- Übersetzen Sie Missbrauchsfälle in konkrete Sicherheitstestfälle, die manuell ausgeführt oder als Teil der Testpipeline automatisiert werden können. Jeder Missbrauchsfall sollte mindestens einen Test produzieren, der den beschriebenen Angriff versucht und verifiziert, dass sich das System dagegen verteidigt.

## Tradeoffs ⇄

> Die Definition von Missbrauchsfällen verschiebt Sicherheitsdenken von reaktivem Patchen zu proaktiver Bedrohungsidentifikation, erfordert aber spezialisiertes Wissen und anhaltenden Aufwand, um relevant zu bleiben.

**Vorteile:**

- Identifiziert systematisch Sicherheitslücken, die konventionelle Anforderungsanalyse übersieht, besonders in Legacy-Systemen, wo Sicherheit keine Designpriorität war.
- Produziert konkrete, testbare Sicherheitsanforderungen, die sich direkt in Sicherheitstestfälle und Abnahmekriterien übersetzen.
- Baut gemeinsames Verständnis von Sicherheitsbedrohungen im Entwicklungsteam auf, was das allgemeine Sicherheitsbewusstsein und die Codequalität verbessert.
- Liefert Dokumentation, die Sicherheitsinvestitionen gegenüber Stakeholdern rechtfertigt, indem realistische Angriffsszenarien und deren potenzielle Auswirkung beschrieben werden.

**Kosten und Risiken:**

- Erfordert Sicherheitsexpertise, die im Team möglicherweise nicht existiert, was externe Berater oder Schulung nötig macht.
- Die Missbrauchsfallanalyse kann eine überwältigende Anzahl von Szenarien produzieren, was sorgfältige Priorisierung erfordert, um sich auf die risikoreichsten Punkte zu konzentrieren.
- Missbrauchsfälle werden veraltet, während sich Angriffstechniken weiterentwickeln, was periodische Überprüfung und Aktualisierung erfordert.
- Ohne entsprechende Sicherheitstests und Behebung liefert Missbrauchsfall-Dokumentation Bewusstsein, aber keinen Schutz.

## How It Could Be

> Die folgenden Szenarien veranschaulichen, wie die Definition von Missbrauchsfällen Sicherheitslücken in Legacy-Systemen aufdeckt.

Ein Legacy-Gesundheitsportal erlaubt es Patienten, ihre medizinischen Akten über eine Weboberfläche einzusehen. Die ursprünglichen Anforderungen spezifizierten nur den legitimen Anwendungsfall: „Patient sieht seine eigenen Akten ein." Ein Missbrauchsfall-Workshop identifiziert mehrere Angreiferszenarien: Ein authentifizierter Patient manipuliert URL-Parameter, um auf die Akten eines anderen Patienten zuzugreifen (unsichere direkte Objektreferenz), ein Angreifer fängt unverschlüsselte API-Antworten mit medizinischen Daten ab, und ein unzufriedener Mitarbeiter nutzt seinen administrativen Zugriff, um Patientenakten in großem Umfang zu exportieren. Das Testen dieser Missbrauchsfälle offenbart, dass das System tatsächlich Aktenzugriff durch Modifikation des Patienten-ID-Parameters in der URL erlaubt, da das Backend keine Autorisierungsprüfung über den initialen Login hinaus durchführt. Dieser Befund führt zur Implementierung zeilenbasierter Zugriffskontrolle, wodurch eine Schwachstelle behoben wird, die acht Jahre lang bestanden hatte.

Eine Legacy-Finanzanwendung verarbeitet Überweisungen durch einen mehrstufigen Workflow. Die Missbrauchsfallanalyse identifiziert, dass ein Angreifer mit Zugriff auf ein niedrigberechtigtes „Betrachter"-Konto potenziell den Überweisungsgenehmigungsprozess manipulieren könnte, indem er abgefangene HTTP-Anfragen eines autorisierten Genehmigers wiederholt. Testen bestätigt, dass das System das Session-Cookie validiert, aber nicht verifiziert, dass der authentifizierte Nutzer Genehmigungsbefugnis für die spezifische Transaktion hat. Der Missbrauchsfall informiert direkt die Behebung: Hinzufügen rollenbasierter Autorisierungsprüfungen bei jedem Schritt des Genehmigungs-Workflows, nicht nur beim initialen Login.
